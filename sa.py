# # Example run
# running outputs of base model V1 and save plots to html

# In[15]:
from __future__ import annotations
from dataclasses import dataclass, field, replace, is_dataclass
from typing import Dict, List, Deque, Any, Tuple, Callable, Optional, Iterable
import math, copy, itertools
from collections import deque
from pathlib import Path

from pathlib import Path
from plotly.io import to_html
import webbrowser

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# from model.py
from model import (
    Config,
    Simulation,
    run_single,
    run_reps,
    summarize_reps,
    run_scenarios,
    scenarios_to_df,
)

class Report:
    """
    Collect Plotly figures and make a single self-contained HTML.
    Optionally also save each figure as its own HTML/PNG.
    """
    def __init__(self, out_dir: Path, *, open_in_browser: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._figs: list[tuple[object, str]] = []
        self._open = bool(open_in_browser)

    def add(self, fig, title: str, *, save_individual: bool = True, png: bool = False):
        """Register a figure and (optionally) save standalone assets."""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in title.lower())
        if save_individual:
            html_path = self.out_dir / f"{safe}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn", auto_open=False)
            print(f"[fig] {html_path}")
            if png:
                try:
                    fig.write_image(str(self.out_dir / f"{safe}.png"), scale=2)  # needs kaleido
                except Exception as e:
                    print(f"[warn] PNG export failed for {title!r}: {e}")
        self._figs.append((fig, title))

    def save(self, filename: str = "report.html", *, heading: str = "Simulation report"):
        """Write one combined HTML file with all figures."""
        body = []
        for fig, title in self._figs:
            body.append(f"<h2>{title}</h2>")
            body.append(to_html(fig, include_plotlyjs=False, full_html=False))
        doc = f"""<!doctype html>
<html><head><meta charset="utf-8" />
<title>{heading}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Inter,sans-serif; margin:24px}}
 h1{{margin:0 0 12px}}
 h2{{margin:28px 0 8px; font-size:18px}}
</style>
</head><body>
<h1>{heading}</h1>
{''.join(body)}
</body></html>"""
        out = self.out_dir / filename
        out.write_text(doc, encoding="utf-8")
        print(f"[report] {out}")
        if self._open:
            try: webbrowser.open_new_tab(out.resolve().as_uri())
            except Exception: pass

###################################################

def main():  

    def clone_cfg(cfg):
        """safe copy for scenario overrides."""
        return copy.deepcopy(cfg)
    
    def deep_update(dst: dict, src: dict) -> dict:
        """Recursive dict merge: dst <- src (modifies dst, returns dst)."""
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst
    
    def apply_overrides(cfg, overrides: Dict[str, Any]):
        """
        Apply scenario overrides onto a cloned Config.
        Supports both top-level attributes and nested dict merges.
        """
        cfg2 = clone_cfg(cfg)
        for k, v in (overrides or {}).items():
            cur = getattr(cfg2, k, None)
            if isinstance(cur, dict) and isinstance(v, dict):
                deep_update(cur, v)  # in-place
            else:
                setattr(cfg2, k, v)
        return cfg2
    
    
    # # Single run of base config
    
    # In[ ]:
    
    
    def run_single(cfg, run_id: int = 1) -> dict:
        sim = Simulation(cfg)
        return sim.single_run(run_id=run_id)
    
    
    # # Mulitple reps of base config
    
    # In[ ]:
    
    
    def run_reps(cfg, n_reps: int) -> Tuple[list, dict]:
        """
        Returns:
          rows: list of KPI dicts (one per rep; each has 'rep').
          results_dict: {rep -> list[patient-record-dicts]}
        """
        sim = Simulation(cfg)
        rows: List[dict] = []
        results_dict: Dict[int, list] = {}
        for r in range(1, int(n_reps) + 1):
            res = sim.single_run(run_id=r)
            rows.append({"rep": r, **res["kpis"]})
            results_dict[r] = res["patients"]
        return rows, results_dict
    
    
    # In[ ]:
    
    
    def summarize_reps(rows):
        try:
            import pandas as pd
            df = pd.json_normalize(rows)
            desc = df.select_dtypes(include="number").describe()
            means = df.mean(numeric_only=True)
            stds  = df.std(numeric_only=True, ddof=1)
            ci95  = 1.96 * stds / (len(df) ** 0.5)
            summary = pd.DataFrame({"mean": means, "std": stds, "ci95": ci95})
            return df, desc, summary
        except Exception:
            return None, None, None
    
    
    # # Build scenarios
    
    # In[ ]:
    
    
    def run_scenarios(
        base_cfg,
        scenarios: Dict[str, Dict[str, Any]],
        n_reps: int,
        *,
        attach_patients_last_only: bool = True,
    ):
        """
        Runs each scenario for n_reps.
        Returns:
          kpi_rows: list of dicts (columns: scenario, rep, KPIs...)
          patients_by_scenario: {scenario -> {rep -> list[patient dicts]}}
             If attach_patients_last_only=True, stores only the last replication's patients per scenario
             under key rep == n_reps (compact for plotting later).
        """
        kpi_rows: List[dict] = []
        patients_by_scenario: Dict[str, Dict[int, list]] = {}
    
        for scen_name, overrides in scenarios.items():
            cfg_s = apply_overrides(base_cfg, overrides)
            cfg_s.trace("Scenario config applied", scenario=scen_name)
            sim = Simulation(cfg_s)
            patients_by_scenario[scen_name] = {}
    
            for r in range(1, int(n_reps) + 1):
                res = sim.single_run(run_id=r)
                kpi_rows.append({"scenario": scen_name, "rep": r, **res["kpis"]})
    
                # store patients either every rep or only last rep
                if not attach_patients_last_only or r == n_reps:
                    patients_by_scenario[scen_name][r] = res["patients"]
    
        return kpi_rows, patients_by_scenario
    
    
    # In[ ]:
    
    
    def scenarios_to_df(kpi_rows):
        try:
            import pandas as pd
            return pd.json_normalize(kpi_rows)
        except Exception:
            return None
    
    
    # --- Base config ---
    cfg = Config()
    cfg.horizon_days = 490 #includes warmup
    cfg.warmup_days  = 140
    cfg.session_minutes = 240
    cfg.trauma_sessions_by_dow = {0:4,1:2,2:4,3:2,4:4,5:2,6:2}
    cfg.iat_mean_per_day = {"hip":360.0, "shoulder":1440.0, "wrist":480.0, "ankle":720.0}
    cfg.duration_params = {
        "hip": {"mean":90.0, "sd":25.0, "min":30.0},
        "shoulder":{"mean":75.0, "sd":22.5, "min":20.0},
        "wrist":{"mean":60.0, "sd":18.0, "min":15.0},
        "ankle":{"mean":45.0, "sd":27.0, "min":20.0},
    }
    cfg.breach_deadline_minutes = {"hip":36*60, "shoulder":14*24*60, "wrist":14*24*60, "ankle":14*24*60}
    cfg.service_policy = "pi"
    cfg.priority_weights = {"hip":1.0,"shoulder":1.0,"wrist":1.0,"ankle":1.0}
    cfg.base_seed = 1984
    cfg.TRACE = False

    # save images to html for sharing
    out_dir = Path("outputs")
    report = Report(out_dir, open_in_browser=True)
    
    # --- Single run ---
    res = run_single(cfg, run_id=1)
    print("KPIs:", list(res["kpis"].keys()))
    
    # --- Reps ---
    # rows, results_dict = run_reps(cfg, n_reps=30)
    # df, desc, summary = summarize_reps(rows)
    # if df is not None:
    #     print("Replication summary:"); print(desc)
    #     print("\nKPI means ±95% CI:"); print(summary)
    
    # --- Scenarios ---
    scenarios = {
      "baseline_pi": {},
      "more_hips_priority": {"priority_weights": {"hip": 2.0}},
      "edd_policy": {"service_policy": "edd"},
      "more_capacity_mon": {"trauma_sessions_by_dow": {0:5,1:2,2:4,3:2,4:4,5:2,6:2}},
      "more_capacity_fri": {"trauma_sessions_by_dow": {0:4,1:2,2:4,3:2,4:5,5:2,6:2}},
    }
    
    kpi_rows, patients_by_scen = run_scenarios(cfg, scenarios, n_reps=300, attach_patients_last_only=False)
    df_scen = scenarios_to_df(kpi_rows)
    if df_scen is not None:
        print("\nScenario KPI head:")
        print(df_scen.head())
    
    
    # # Plots
    
    # ## KPI plots
    
    # ### Helpers - reshape and CIs
    
    def patients_to_df(patients_by_scen: Dict[str, Dict[int, list]]) -> pd.DataFrame:
        frames = []
        for scen, rep_map in patients_by_scen.items():
            for rep, plist in rep_map.items():
                if not plist:
                    continue
                d = pd.DataFrame(plist)
                if d.empty:
                    continue
                d["scenario"] = scen
                d["rep"] = int(rep)
                frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    
    # after kpi_rows, patients_by_scen = run_scenarios(...)
    patients_df = patients_to_df(patients_by_scen).drop_duplicates(subset=["scenario","rep","id"])
    
    warmup_cut_min = cfg.get_warmup_cut_min()
    end_time_min   = cfg.get_horizon_end_min()

    # confirm each scenario actually changed what you think
    for name, ov in scenarios.items():
        c = apply_overrides(cfg, ov)
        print(f"\n{name}")
        print(" policy:", c.get_policy(), "weights:", c.get_priority_weights()["hip"],
              "Sessions:", c.get_sessions_by_dow().get(0))

    def add_plot(fig, caption):
        # Centralize how to push figs into the report
        report.add(fig, caption)
        # If also want live window locally, uncomment:
        # fig.show()
    
    #### HELPERS - reshape and CIs
    
    # === place these where your helpers currently are ===
    def _is_scalar_series(s: pd.Series) -> bool:
        return (
            pd.api.types.is_number(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)
        ) and not s.apply(lambda x: isinstance(x, (dict, list, tuple, np.ndarray))).any()
    
    def melt_kpis(
        df_scen: pd.DataFrame,
        kpis: list[str] | None = None,
        *,
        id_cols: tuple[str, ...] = ("scenario", "rep"),
        prefixes: tuple[str, ...] | None = None,
        exclude_prefixes: tuple[str, ...] | None = ("utilisation_by_day_",),
        scenario_order: list[str] | None = None,
        value_name: str = "value",
        dropna: bool = True,
        coerce_numeric: bool = False,
    ) -> pd.DataFrame:
        df = df_scen.copy()
        present_ids = [c for c in id_cols if c in df.columns]
        if kpis is None:
            candidate_cols = [c for c in df.columns if c not in present_ids]
            if exclude_prefixes:
                candidate_cols = [c for c in candidate_cols if not any(c.startswith(p) for p in exclude_prefixes)]
            kpis = [c for c in candidate_cols if _is_scalar_series(df[c])]
        if prefixes:
            pref_cols = [c for c in df.columns for p in prefixes if c.startswith(p)]
            kpis = sorted(set(kpis).union(pref_cols))
        missing = [c for c in kpis if c not in df.columns]
        if missing:
            raise KeyError(f"Requested KPI(s) not in DataFrame: {missing}")
        long = df.melt(id_vars=present_ids, value_vars=kpis, var_name="kpi", value_name=value_name)
        if coerce_numeric:
            long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
        if dropna:
            long = long[long[value_name].notna()].copy()
        if "scenario" in long.columns:
            if scenario_order is None:
                scenario_order = pd.unique(long["scenario"])
            long["scenario"] = pd.Categorical(long["scenario"], categories=list(scenario_order), ordered=True)
        return long
    
    def ci95(
        df_long: pd.DataFrame,
        *,
        val: str = "value",
        group_cols: tuple[str, ...] = ("scenario", "kpi"),
        z: float = 1.96,
        observed: bool = True,
    ) -> pd.DataFrame:
        agg = (df_long.groupby(list(group_cols), as_index=False, observed=observed)
                      .agg(mean=(val, "mean"), n=(val, "size"), sd=(val, "std")))
        agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
        agg["lo"] = agg["mean"] - z * agg["se"]
        agg["hi"] = agg["mean"] + z * agg["se"]
        return agg



    #### kpi bars plus CIs (egs)

    long = melt_kpis(
        df_scen,
        prefixes=("breach_incidence_total",
                  "pct_within_deadline_overall",
                  "waiting_count_final",
                  "utilisation_rate_post_warmup")
    )
    summ = ci95(long)
    
    # Breach incidence CI bar
    to_plot = summ[summ["kpi"].eq("breach_incidence_total")]
    fig_breach_ci = px.bar(
        to_plot, x="scenario", y="mean",
        error_y=to_plot["hi"]-to_plot["mean"],
        error_y_minus=to_plot["mean"]-to_plot["lo"],
        title="Breach incidence (mean ±95% CI)"
    )
    #add_plot(fig_breach_ci, "Breach incidence — mean ±95% CI")
    
    # Faceted CI bars for all selected KPIs
    fig_kpis_ci = px.bar(
        summ, x="scenario", y="mean", color="scenario",
        facet_col="kpi", facet_col_wrap=2,
        error_y=summ["hi"]-summ["mean"], error_y_minus=summ["mean"]-summ["lo"],
        title="KPIs by scenario (mean ±95% CI)"
    )
    fig_kpis_ci.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    #add_plot(fig_kpis_ci, "KPIs by scenario — mean ±95% CI")

    ################################### throughput and breaches by group

    AMB_KINDS = ("shoulder", "wrist", "ankle")

    def make_grouped_long(df_scen: pd.DataFrame, amb_kinds=AMB_KINDS) -> pd.DataFrame:
        df = df_scen.copy()
        breach_prefixes = ("breach_incidence_by_kind.","breached_by_kind.","breached_served_by_kind.")
        def get_breach_val(row, kind):
            for pref in breach_prefixes:
                c = f"{pref}{kind}"
                if c in row and pd.notna(row[c]):
                    return row[c]
            return np.nan
    
        recs = []
        for _, r in df.iterrows():
            hip_served = r.get("served_by_kind.hip", np.nan)
            amb_served = np.nansum([r.get(f"served_by_kind.{k}", np.nan) for k in amb_kinds])
            hip_breach = get_breach_val(r, "hip")
            amb_breach = np.nansum([get_breach_val(r, k) for k in amb_kinds])
    
            for group, served, breached in (
                ("hip", hip_served, hip_breach),
                ("ambulatory", amb_served, amb_breach),
            ):
                recs.append({"scenario": r["scenario"], "rep": r["rep"], "group": group, "kpi": "served_total", "value": served})
                recs.append({"scenario": r["scenario"], "rep": r["rep"], "group": group, "kpi": "breached_total", "value": breached})
        return pd.DataFrame.from_records(recs)

    
    long_grp = make_grouped_long(df_scen)
    fig_grp = px.box(
        long_grp, x="scenario", y="value", color="group",
        facet_col="kpi", facet_col_wrap=2, points="all",
        title="Throughput & Breaches by Group (per replication)"
    )
    fig_grp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_grp.update_yaxes(matches=None, rangemode="tozero")
    add_plot(fig_grp, "Throughput & breaches — hip vs ambulatory")

    ###################### breach snapshots by kind 

    cols = [c for c in df_scen.columns if c.startswith((
        "in_breach_start_total_by_kind.",
        "in_breach_end_total_by_kind.",
        "new_breaches_total_by_kind."
    ))]
    long_snap = melt_kpis(df_scen, cols).copy()
    long_snap["metric"] = (long_snap["kpi"]
        .str.replace("_total_by_kind.", " ", regex=False)
        .str.replace("in_breach_", "in-", regex=False)
        .str.replace("new_", "new ", regex=False))
    long_snap["kind"] = long_snap["kpi"].str.split(".").str[-1]
    
    fig_snap = px.box(
        long_snap, x="scenario", y="value", color="scenario",
        facet_row="metric", facet_col="kind", facet_col_wrap=4, points="all",
        title="Breach snapshots (summed post-warmup)"
    )
    fig_snap.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_snap.update_yaxes(matches=None, rangemode="tozero")
    #add_plot(fig_snap, "Breach snapshots — start / end / new by kind")

    ############# wait times (served patients)

    dfp = patients_df[patients_df["service_start"].notna()].copy()
    dfp["cat"] = np.where(dfp["kind"].eq("hip"), "hip", "amb")
    dfp["_wait_days"] = dfp["wait"].astype(float) / 1440.0
    
    fig_wait = px.violin(
        dfp, x="scenario", y="_wait_days", color="cat",
        box=True, points=False,
        labels={"_wait_days":"Wait (days)","cat":"Group"},
        title="Wait time of served patients (post-warmup)"
    )
    fig_wait.update_yaxes(rangemode="tozero")
    add_plot(fig_wait, "Wait time (served) — hip vs ambulatory")

    ############# cumulative servied by day

    served = (dfp.assign(service_day=(dfp["service_start"]//1440).astype(int))
                  .groupby(["scenario","service_day"], as_index=False, observed=True)
                  .size().rename(columns={"size":"served"}))
    served["cum"] = served.groupby("scenario", observed=True)["served"].cumsum()
    
    fig_cum_serv = px.line(
        served, x="service_day", y="cum", color="scenario",
        labels={"service_day":"Day","cum":"Cumulative served"},
        title="Cumulative served patients by day"
    )
    add_plot(fig_cum_serv, "Cumulative served by day")

    ################# served/breached by weekday

    WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    def make_weekday_long(patients_by_scen: dict, cfg) -> pd.DataFrame:
        """
        One row per (scenario, rep, group, weekday_name, kpi), with kpi in {"served_total", "breached_total"}.
        Counts are post-warmup and before horizon end.
        """

        # Build the per-patient frame in this scope
        pdf = patients_to_df(patients_by_scen).drop_duplicates(subset=["scenario","rep","id"]).copy()

        # Make sure breach_time exists where needed (breached & missing)
        if "breach_time" not in pdf.columns:
            pdf["breach_time"] = pd.NA
        need_bt = pdf["breach_time"].isna() & pdf["breached"].astype("boolean", errors="ignore").fillna(False)
        pdf.loc[need_bt, "breach_time"] = (
            pdf.loc[need_bt, "arrival"].astype("Int64") + pdf.loc[need_bt, "deadline"].astype("Int64")
        )

        warm = cfg.get_warmup_cut_min()
        end  = cfg.get_horizon_end_min()
        pdf["group"] = np.where(pdf["kind"].eq("hip"), "hip", "ambulatory")

        recs: list[pd.DataFrame] = []

        # --- served by weekday (service_start within horizon, post-warmup)
        served = pdf[
            (pdf["service_start"].notna()) &
            (pd.to_numeric(pdf["service_start"], errors="coerce") >= warm) &
            (pd.to_numeric(pdf["service_start"], errors="coerce") <  end)
        ].copy()
        served["weekday_name"] = ((pd.to_numeric(served["service_start"], errors="coerce") // 1440) % 7).astype(int).map(dict(enumerate(WEEKDAYS)))

        g_served = (served.groupby(["scenario","rep","group","weekday_name"], observed=True)
                           .size().rename("value").reset_index())
        g_served["kpi"] = "served_total"
        recs.append(g_served)

        # --- breach incidence by weekday (breach_time within horizon, still waiting at breach)
        breach = pdf[
            (pdf["breach_time"].notna()) &
            (pd.to_numeric(pdf["breach_time"], errors="coerce") >= warm) &
            (pd.to_numeric(pdf["breach_time"], errors="coerce") <  end)
        ].copy()

        ss = pd.to_numeric(breach["service_start"], errors="coerce")
        bt = pd.to_numeric(breach["breach_time"],  errors="coerce")
        still_wait = ss.isna() | (ss > bt)
        breach = breach.loc[still_wait].copy()

        breach["weekday_name"] = ((pd.to_numeric(breach["breach_time"], errors="coerce") // 1440) % 7).astype(int).map(dict(enumerate(WEEKDAYS)))

        g_breach = (breach.groupby(["scenario","rep","group","weekday_name"], observed=True)
                           .size().rename("value").reset_index())
        g_breach["kpi"] = "breached_total"
        recs.append(g_breach)

        # Output
        out = pd.concat(recs, ignore_index=True)
        out["weekday_name"] = pd.Categorical(out["weekday_name"], categories=WEEKDAYS, ordered=True)
        return out


    def ci95_tbl(df, val="value", group_cols=("scenario","group","kpi","weekday_name")):
        g = df.groupby(list(group_cols), as_index=False, observed=True)
        out = g.agg(mean=(val,"mean"), n=(val,"size"), sd=(val,"std"))
        out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
        out["lo"] = out["mean"] - 1.96*out["se"]
        out["hi"] = out["mean"] + 1.96*out["se"]
        return out
    
    weekday_long = make_weekday_long(patients_by_scen, cfg)
    agg_wd = ci95_tbl(weekday_long)
    
    fig_weekday = px.line(
        agg_wd, x="weekday_name", y="mean", color="scenario",
        line_dash="group", facet_col="kpi", facet_col_wrap=2,
        error_y=agg_wd["hi"]-agg_wd["mean"], error_y_minus=agg_wd["mean"]-agg_wd["lo"],
        markers=True,
        title="Served & Breach Incidence by Weekday (mean ±95% CI)"
    )
    fig_weekday.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_weekday.update_yaxes(matches=None, rangemode="tozero")
    add_plot(fig_weekday, "Weekday served & breaches — mean ±95% CI")

    ################## weekday breaches (means)

    for kpi in ["served_total", "breached_total"]:
        heat_df = (
            agg_wd[agg_wd["kpi"].eq(kpi)]
            .pivot_table(
                index=["scenario", "group"],
                columns="weekday_name",
                values="mean",
                observed=False
            )
            .reindex(columns=WEEKDAYS)
        )
        labels = [f"{s} • {g}" for s, g in heat_df.index]
        fig_heat = px.imshow(
            heat_df.values,
            x=heat_df.columns,
            y=labels,
            labels=dict(x="Weekday", y="Scenario • Group", color=f"mean {kpi}"),
            text_auto=True,
            aspect="auto",
            title=f"{kpi}: mean by weekday"
        )
        fig_heat.update_coloraxes(colorscale="Viridis", cmin=0)
        add_plot(fig_heat, f"{kpi} — mean by weekday (heatmap)")

    ############# weekday bars

    for kpi in ["served_total", "breached_total"]:
        dat = agg_wd[agg_wd["kpi"].eq(kpi)].copy()
        dat["errp"] = (dat["hi"] - dat["mean"]).fillna(0)
        dat["errm"] = np.minimum(dat["mean"] - dat["lo"], dat["mean"]).fillna(0)
    
        fig_bar_wd = px.bar(
            dat,
            x="weekday_name",
            y="mean",
            color="scenario",
            facet_row="group",
            barmode="group",
            error_y="errp",
            error_y_minus="errm",
            category_orders={"weekday_name": WEEKDAYS},
            title=f"{kpi}: mean ±95% CI by weekday"
        )
        fig_bar_wd.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_bar_wd.update_yaxes(matches=None, range=[0, None])
        fig_bar_wd.update_traces(error_y=dict(symmetric=False, width=4, thickness=1))
        add_plot(fig_bar_wd, f"{kpi} — weekday means ±95% CI")

    ############# backlog at horizon end

    long_backlog = melt_kpis(df_scen, ["waiting_count_final","breached_waiting_final"])
    fig_backlog = px.box(
        long_backlog, x="scenario", y="value", color="scenario",
        facet_col="kpi", facet_col_wrap=2,
        title="End-of-horizon backlog & breached backlog"
    )
    fig_backlog.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_backlog.update_yaxes(matches=None, rangemode="tozero")
    add_plot(fig_backlog, "Backlog at horizon end")

    #### utilisation

    fig_util_rate = px.box(
        melt_kpis(df_scen, ["utilisation_rate_post_warmup"]),
        x="scenario", y="value", color="scenario",
        title="Utilisation rate (post-warmup)"
    )
    fig_util_rate.update_yaxes(tickformat=".0%")
    add_plot(fig_util_rate, "Utilisation rate (post-warmup)")

    ##### utilisation time series

    u = (df_scen[["scenario","rep","utilisation_by_day_post_warmup"]]
       .explode("utilisation_by_day_post_warmup").dropna())
    u = pd.concat([u.drop(columns=["utilisation_by_day_post_warmup"]),
                   u["utilisation_by_day_post_warmup"].apply(pd.Series)], axis=1)
    g = (u.groupby(["scenario","day"], as_index=False)
           .agg(mean=("utilisation","mean"), n=("utilisation","size"), se=("utilisation","sem")))
    g["lo"] = (g["mean"] - 1.96*g["se"]).clip(lower=0)
    g["hi"] = (g["mean"] + 1.96*g["se"]).clip(upper=1)
    
    fig_util_ts = px.line(g, x="day", y="mean", color="scenario",
                          title="Utilisation over time (mean ±95% CI)")
    # (Add invisible hi/lo traces just for hover if you like; optional)
    add_plot(fig_util_ts, "Utilisation over time — mean ±95% CI")


    ####### throughput vs breach

    def _pick_breach_y(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        # 1) Prefer event-based incidence if present
        if "breach_incidence_total" in df.columns:
            return df, "breach_incidence_total"
        # 2) Fall back to served-in-breach total if present
        if "breached_total" in df.columns:
            return df, "breached_total"
        # 3) Otherwise, derive from by-kind columns if available
        kind_breached = [c for c in df.columns if c.startswith("breached_by_kind.")]
        if kind_breached:
            df = df.copy()
            df["breached_total"] = df[kind_breached].sum(axis=1, skipna=True)
            return df, "breached_total"
        kind_incidence = [c for c in df.columns if c.startswith("new_breaches_total_by_kind.")]
        if kind_incidence:
            df = df.copy()
            df["breach_incidence_total"] = df[kind_incidence].sum(axis=1, skipna=True)
            return df, "breach_incidence_total"
        # If truly nothing found, raise a clear error
        raise KeyError(
            "No breach total available. Expected one of: "
            "'breach_incidence_total', 'breached_total', "
            "'breached_by_kind.*', or 'new_breaches_total_by_kind.*'."
        )
    
    df_scatter, breach_y = _pick_breach_y(df_scen)
    
    fig_scatter = px.scatter(
        df_scatter,
        x="served_total",
        y=breach_y,
        color="scenario",
        hover_data=["rep"],
        trendline="ols",
        trendline_scope="trace",
        title="Throughput vs Breaches (per replication)",
        labels={"served_total": "Throughput (served)", breach_y: "Breaches"},
    )
    add_plot(fig_scatter, "Throughput vs breaches — per replication")


    ################# cumulative breaches


    warmup_cut_min = cfg.get_warmup_cut_min()
    end_time_min   = cfg.get_horizon_end_min()

    dfb = patients_df.copy()
    if "breach_time" not in dfb.columns:
        dfb["breach_time"] = pd.NA

    need_bt = dfb["breach_time"].isna() & dfb["breached"].astype("boolean", errors="ignore").fillna(False)
    dfb.loc[need_bt, "breach_time"] = dfb["arrival"].astype("Int64") + dfb["deadline"].astype("Int64")

    # Ensure numeric and build NA-safe mask on dfb (not 'breach')
    ss = pd.to_numeric(dfb["service_start"], errors="coerce")
    bt = pd.to_numeric(dfb["breach_time"],  errors="coerce")
    still_wait = (ss.isna() | (ss > bt).fillna(False))

    mask_event = bt.notna() & still_wait & (bt >= warmup_cut_min) & (bt < end_time_min)

    events = dfb.loc[mask_event, ["scenario","rep","kind","breach_time"]].copy()
    events["breach_day"] = (events["breach_time"] // 1440).astype(int)

    daily = (events.groupby(["scenario","breach_day"], observed=True).size()
             .rename("breaches").reset_index().sort_values(["scenario","breach_day"]))
    daily["cum"] = daily.groupby("scenario", observed=True)["breaches"].cumsum()

    fig_cum_breach = px.line(
        daily, x="breach_day", y="cum", color="scenario",
        title="Cumulative breach events by day (post-warmup)",
        labels={"breach_day":"Day", "cum":"Cumulative breaches"}
    )
    add_plot(fig_cum_breach, "Cumulative breach events — overall")

    events["group"] = np.where(events["kind"].eq("hip"), "hip", "amb")
    daily_g = (events.groupby(["scenario","group","breach_day"], observed=True).size()
               .rename("breaches").reset_index().sort_values(["scenario","group","breach_day"]))
    daily_g["cum"] = daily_g.groupby(["scenario","group"], observed=True)["breaches"].cumsum()

    fig_cum_breach_g = px.line(
        daily_g, x="breach_day", y="cum", color="group",
        facet_col="scenario", facet_col_wrap=2, markers=True,
        title="Cumulative breach events — hip vs ambulatory",
        labels={"breach_day":"Day","cum":"Cumulative breaches","group":"Group"}
    )
    fig_cum_breach_g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    add_plot(fig_cum_breach_g, "Cumulative breach events — hip vs ambulatory")


    report.save("report.html", heading="Scenario analysis — KPIs & patient-level views")

if __name__ == "__main__":
    main()