#!/usr/bin/env python
# coding: utf-8

# # Scheduling model for hips and ambulatory surgery
# 
# Patients with hip fractures and ambulatory fractures arrive into the model. They have surgical deadlines applied. Patients are scheduled according to a 'slack' rule divided by a weighting, which is defined as:
# 
# Time to deadline / weight.
# 
# Hips are prioritised over ambulatory patients, but 'slack' ensures that ambulatory patients get seen.
# 
# An optional 'duration' parameter will try to schedule longer patients in first.
# 
# > Question: do we need an end of day 'packer' to fit short cases into remaining time? Done as utilisation was low in many cases.
# > 
# > Question: breaches for served patients occur when their wait at service start exceeds deadline. This might need to change so that breaches are counted at the time of breaching. 

# In[ ]:


from __future__ import annotations
from dataclasses import dataclass, field, replace, is_dataclass
from typing import Dict, List, Deque, Any, Tuple, Callable, Optional, Iterable
import math, copy, itertools

from collections import deque, defaultdict
from pathlib import Path
from sim_tools.distributions import Exponential, Normal

import numpy as np
import pandas as pd
import plotly.express as px




# ## `Config` contains all the parameters used to run the model

# In[ ]:


AMB_KINDS = ("shoulder", "wrist", "ankle")
ALL_KINDS = ("hip",) + AMB_KINDS  # ("hip", "shoulder", "wrist", "ankle")

@dataclass
class Config:

    horizon_days: int = 182   # around 6 mths 
    base_seed: int = 1984
    warmup_days: int = 21 # allow build-up of amb patients

    # ================= Capacity / arrivals =================
    # Number of sessions per day. 0 = Monday
    # Each instance gets its own new dictionary
    trauma_sessions_by_dow: Dict[int, float] = field(
        default_factory=lambda: {0: 4.0, 1: 2.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 2.0, 6: 2.0}
    )
    session_minutes: int = 240 #ie half day (minutes)
    # interarrival times (minutes)
    iat_mean_per_day: Dict[str, float] = field(
        default_factory=lambda: {
            "hip": 360.0,
            "shoulder": 7200.0,
            "wrist": 1440.0,
            "ankle": 720.0,
        }
    )

    # ================= Case durations =================
    # mean, sd and min values (minutes)
    duration_params: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "hip":      {"mean": 120.0, "sd": 54.0, "min": 30.0},
            "shoulder": {"mean": 55.0,  "sd": 22.5, "min": 20.0},
            "wrist":    {"mean": 40.0,  "sd": 18.0, "min": 15.0},
            "ankle":    {"mean": 40.0,  "sd": 27.0, "min": 20.0},
        }
    )

    # ================= Breach deadlines =================
    # 36 hours for hips, 14 days for ambulatory cases
    breach_deadline_minutes: Dict[str, int] = field(
        default_factory=lambda: {
            "hip": 36 * 60,
            "shoulder": 14 * 24 * 60,
            "wrist": 14 * 24 * 60,
            "ankle": 14 * 24 * 60,
        }
    )

    # ================= Service policy =================
    # Up for discussion - edd is just weighted slack (time to deadline); 
    # pi includes expected (not sampled) duration - weighs longer surgeries as more urgent
    # hip can get a higher weight (denominator) so higher priority (lower edd/pi)
    # start with no weights at all
    service_policy: str = "edd"  # "edd" or "pi"
    priority_weights: Dict[str, float] = field(
        default_factory=lambda: {"hip": 2.0, "shoulder": 1.0, "wrist": 1.0, "ankle": 1.0}
    )

    # ================= utils =======================
    TRACE: bool = False

    def trace(self, *args, **kwargs) -> None:
        if not getattr(self, "TRACE", False): return
        msg = " ".join(str(a) for a in args)
        if kwargs:
            details = " ".join(f"{k}={v!r}" for k,v in kwargs.items())
            print(f"[TRACE] {msg} {details}")
        else:
            print(f"[TRACE] {msg}")


    #################################################
    # ================= VALIDATION  =================
    #################################################


    def validate(self, debug: bool = False) -> None:
        """
        Validate configuration for the theatre scheduling simulation.

        Args:
            debug (bool): If True, prints detailed debug information at each step.

        Raises:
            ValueError: if validation fails.
        """
        errors = []
        warnings = []

        def dbg(msg):
            if debug:
                print(f"[validate] {msg}")

        def err(msg):
            errors.append(str(msg))
            if debug:
                print(f"[ERROR] {msg}")

        def warn(msg):
            warnings.append(str(msg))
            if debug:
                print(f"[WARN] {msg}")

        dbg("Starting config validation...")

        # -------------------- basic scalars --------------------
        dbg("Checking warmup_days and horizon_days...")
        warmup_days  = self.get_warmup_days()
        horizon_days = self.get_horizon_days()
        if warmup_days < 0:
            err("warmup_days must be >= 0.")
        if horizon_days <= 0:
            err("horizon_days must be > 0.")
        if horizon_days and warmup_days >= horizon_days:
            err(f"warmup_days ({warmup_days}) must be < horizon_days ({horizon_days}).")
        dbg(f"warmup_days={warmup_days}, horizon_days={horizon_days}")

        # base_seed
        base_seed = self.get_base_seed()
        try:
            base_seed = int(base_seed)
            dbg(f"base_seed={base_seed}")
        except Exception:
            err(f"base_seed must be castable to int, got: {base_seed!r}")

        # policy
        policy = self.get_policy()
        dbg(f"service_policy={policy}")
        if policy not in {"edd", "pi"}:
            err(f"service_policy must be 'edd' or 'pi', got: {policy!r}")

        # priority_weights
        dbg("Checking priority_weights...")
        pweights = self.get_priority_weights()
        if not isinstance(pweights, dict):
            err("priority_weights must be a dict of {kind: positive number}.")
        else:
            for k, v in pweights.items():
                try:
                    fv = float(v)
                    dbg(f"priority_weights[{k}]={fv}")
                    if fv <= 0:
                        err(f"priority_weights[{k!r}] must be > 0, got {v!r}")
                except Exception:
                    err(f"priority_weights[{k!r}] must be numeric, got {v!r}")

        # -------------------- sessions / capacity --------------------
        dbg("Checking session structure and minutes...")
        try:
            sessions_by_dow = self.get_sessions_by_dow()
            dbg(f"sessions_by_dow={sessions_by_dow}")
            for dow, n in sessions_by_dow.items():
                fn = float(n)
                if fn < 0:
                    err(f"sessions_by_dow[{dow}] must be >= 0, got {n!r}")
        except Exception as e:
            err(f"get_sessions_by_dow() failed: {e}")
            sessions_by_dow = {}

        try:
            session_minutes = self.get_session_minutes()
            if callable(session_minutes):
                val = float(session_minutes(0))
            else:
                val = float(session_minutes)
            dbg(f"session_minutes example={val}")
            if val <= 0:
                err(f"session_minutes must be > 0, got {val!r}")
        except Exception as e:
            err(f"get_session_minutes() failed: {e}")
            session_minutes = 0

        # quick weekly capacity sanity check
        try:
            sm = float(session_minutes(0)) if callable(session_minutes) else float(session_minutes)
            total_weekly_capacity = sum(float(sessions_by_dow.get(d, 0.0)) * sm for d in range(7))
            dbg(f"total_weekly_capacity={total_weekly_capacity}")
            if total_weekly_capacity == 0:
                warn("Total weekly capacity is 0 minutes (all sessions_by_dow are zero).")
        except Exception as e:
            warn(f"Could not compute total weekly capacity: {e}")

        # -------------------- clinical models / kinds alignment --------------------
        dbg("Validating duration model...")
        try:
            duration_model = self.get_duration_model()
            dbg(f"Duration kinds={list(duration_model.keys())}")
            for kind, params in duration_model.items():
                dbg(f"Checking duration for kind={kind}: {params}")
                m = float(params.get("mean", 0))
                s = float(params.get("sd", 0))
                if m <= 0:
                    err(f"Duration mean for {kind!r} must be > 0, got {m!r}")
                if s < 0:
                    err(f"Duration sd for {kind!r} must be >= 0, got {s!r}")
        except Exception as e:
            err(f"get_duration_model() failed: {e}")
            duration_model = {}

        dbg("Validating breach deadlines...")
        try:
            deadlines_min = self.get_breach_deadlines_minutes()
            dbg(f"Deadlines={deadlines_min}")
            for kind, mins in deadlines_min.items():
                dm = int(mins)
                if dm <= 0:
                    err(f"Deadline for {kind!r} must be > 0 minutes, got {mins!r}")
        except Exception as e:
            err(f"get_breach_deadlines_minutes() failed: {e}")
            deadlines_min = {}

        # -------------------- arrival model --------------------
        dbg("Validating arrivals (iat_mean_per_day)...")
        try:
            arrivals = self.get_iat_mean_per_day()  # {kind: float minutes}
            dbg(f"Arrival kinds={list(arrivals.keys())}")
            for kind, iat in arrivals.items():
                try:
                    f = float(iat)
                    if f <= 0:
                        err(f"iat_mean_per_day for '{kind}' must be > 0 minutes, got {iat!r}")
                except Exception:
                    err(f"iat_mean_per_day for '{kind}' must be numeric, got {iat!r}")
        except Exception as e:
            err(f"get_iat_mean_per_day() failed: {e}")
            arrivals = {}

        # -------------------- kinds alignment --------------------
        dbg("Checking kind alignment between models...")
        kinds_from_dur = set(duration_model.keys())
        kinds_from_dead = set(deadlines_min.keys())
        kinds_from_arr = set(arrivals.keys()) if isinstance(arrivals, dict) else set()

        missing_in_dead = kinds_from_dur - kinds_from_dead
        missing_in_dur  = kinds_from_dead - kinds_from_dur
        if missing_in_dead:
            err(f"Deadlines missing kinds present in duration model: {sorted(missing_in_dead)}")
        if missing_in_dur:
            err(f"Duration model missing kinds present in deadlines: {sorted(missing_in_dur)}")

        if kinds_from_arr:
            for k in kinds_from_arr - kinds_from_dur:
                warn(f"Arrivals define kind {k!r} not present in duration model.")
            for k in kinds_from_arr - kinds_from_dead:
                warn(f"Arrivals define kind {k!r} not present in deadlines mapping.")

        # -------------------- sampler smoke test --------------------

        dbg("Mean minutes sanity check...")
        try:
            means = self.get_mean_case_minutes()  # deterministic, no RNG
            for k, m in means.items():
                if not (m > 0 and m < 1e6):
                    warn(f"Mean case minutes for '{k}' looks suspicious: {m}")
            dbg(f"mean_case_minutes={means}")
        except Exception as e:
            warn(f"Mean minutes check failed: {e}")

        # -------------------- wrap up --------------------
        if errors:
            print("\n❌ CONFIG VALIDATION FAILED ❌")
            for e in errors:
                print("  -", e)
            raise ValueError(f"{len(errors)} error(s) during config validation.")
        else:
            print("✅ Config validation passed without critical errors.")

        if warnings:
            print("\n⚠️ WARNINGS:")
            for w in warnings:
                print("  -", w)

        if debug:
            print("[validate] Completed successfully.")

    ###########################################
    # ----------------------------- Getters ------
    ############################################

    #constants
    def get_horizon_days(self) -> int:
        return int(self.horizon_days)

    def get_warmup_days(self) -> int:
        # pick one canonical field name and stick to it
        return int(getattr(self, "warmup_days", 0))

    def get_warmup_cut_min(self) -> int:
        return self.get_warmup_days() * 1440

    def get_horizon_end_min(self) -> int:
        return self.get_horizon_days() * 1440

    # surgical kinds
    def get_kinds(self) -> list[str]:
        """Canonical kind list used across models."""
        return sorted(self.get_duration_model().keys())

    #policy and weights
    def get_policy(self) -> str:
        p = str(getattr(self, "service_policy", "edd")).lower()
        return p if p in ("edd","pi") else "edd"

    def get_priority_weights(self) -> dict[str, float]:
        # normalized, positive (>0), missing -> 1.0
        kinds = self.get_kinds()
        raw = dict(getattr(self, "priority_weights", {}) or {})
        out = {}
        for k in kinds:
            v = float(raw.get(k, 1.0))
            out[k] = v if v > 0 else 1.0
        return out

    def get_weight_for(self, kind: str) -> float:
        return self.get_priority_weights().get(str(kind), 1.0)

    #Durations

    def get_duration_model(self) -> Dict[str, dict]:
        """
        Return duration specs per kind directly from cfg.duration_params.
        Each entry is {kind: {"mean": float, "sd": float, "min": float}}.
        """
        out: Dict[str, dict] = {}
        for kind, p in self.duration_params.items():
            m  = float(p["mean"])
            sd = float(p["sd"])
            mn = float(p.get("min", 0.0))
            out[str(kind)] = {"mean": m, "sd": sd, "min": mn}
        return out

    def get_mean_case_minutes(self) -> dict[str, float]:
        dm = self.get_duration_model()
        return {k: float(v["mean"]) for k, v in dm.items()}


    #Deadlines
    def get_breach_deadlines_minutes(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.breach_deadline_minutes.items()}

    def get_deadline_for(self, kind: str) -> int:
        return int(self.get_breach_deadlines_minutes()[str(kind)])

    #Arrivals
    def get_iat_mean_per_day(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.iat_mean_per_day.items()}

    def get_iat_for(self, kind: str) -> float:
        return float(self.get_iat_mean_per_day()[str(kind)])

    def get_arrival_rate_per_min(self) -> dict[str, float]:
        # convenience: lambda per-minute rate = 1 / mean interarrival minutes
        iat = self.get_iat_mean_per_day()
        return {k: (1.0 / v if v > 0 else 0.0) for k, v in iat.items()}

    # Capacity, calendar

    def get_sessions_by_dow(self) -> dict[int, float]:
        return {int(k): float(v) for k, v in self.trauma_sessions_by_dow.items()}

    def get_session_minutes(self):
        # if you later support a callable, keep this as-is and branch at call site
        return int(self.session_minutes)

    def capacity_minutes_for_day(self, day: int) -> float:
        dow = day % 7
        sessions_today = float(self.get_sessions_by_dow().get(dow, 0.0))
        sm = self.get_session_minutes()
        m_per_session = float(sm(day)) if callable(sm) else float(sm)
        return sessions_today * m_per_session

    def get_total_weekly_capacity_minutes(self) -> float:
        sm = self.get_session_minutes()
        m_per_session = float(sm(0)) if callable(sm) else float(sm)
        return sum(float(self.get_sessions_by_dow().get(d, 0.0)) * m_per_session for d in range(7))

    # seeds
    def get_base_seed(self) -> int:
        return int(getattr(self, "base_seed", 0) or 0)

    def seed_for(self, run_id: int, *, stream: str = "") -> int:
        # deterministic, per-stream seeds (e.g., "arrivals", "durations")
        # avoids cross-correlation between processes
        tag = (hash(stream) & 0x7fffffff)
        return self.get_base_seed() + int(run_id) + tag




# In[ ]:


## Test validation


# In[ ]:


cfg = Config()
cfg.validate(debug=True)

print(cfg.get_kinds())
print(cfg.get_sessions_by_dow())
print(cfg.get_session_minutes())
print(cfg.get_iat_mean_per_day())
print(cfg.get_duration_model())
print(cfg.get_breach_deadlines_minutes())
print(cfg.get_mean_case_minutes())
print(cfg.capacity_minutes_for_day(0))


# ## Daily planner
# 
# Two simple scheduling rules are used 

# In[ ]:


class DailyServer:
    """
    Schedules patients for a single day using a priority policy ("edd" or "pi"),
    with dynamic (current-time) slack, excluding not-yet-arrived patients.
    """

    def __init__(self, *, trace: Callable[..., None] = lambda *a, **k: None):
        self.trace = trace

    def serve_day(
        self,
        *,
        queues: Dict[str, Deque[Tuple[int, int]]],           # per kind: (arrival_min, patient_id)
        minutes_budget: float,                                # operating minutes available today
        day_start_min: int,
        policy: str,                                          # "edd" or "pi"
        deadlines_min: Dict[str, int],                        # per-kind breach deadlines (minutes)
        duration_sampler: Dict[str, Callable[[], float]],     # kind -> sampler() -> duration (minutes)
        mean_case_minutes: Dict[str, float],                  # kind -> E[duration] (for PI index)
        priority_weights: Dict[str, float] | None = None,     # higher weight => higher priority
        prev_day_start_min: int | None = None,                # to compute daily breach incidence
    ) -> Tuple[
        float,                              # minutes_left_today
        Dict[str, int],                     # served_today
        Dict[str, int],                     # breached_today (served while in breach)
        Dict[str, List[int]],               # waits_today (true wait to service start)
        Dict[str, List[int]],               # excess_today (minutes over deadline)
        List[Dict[str, float | int | str | bool]],  # patient_records
        Dict[str, int],                     # in_breach_start
        Dict[str, int],                     # new_breaches_today
    ]:
        tr = self.trace
        pol = (policy or "edd").lower()
        if pol not in ("edd", "pi"):
            raise ValueError(f"Unknown policy '{policy}'. Use 'edd' or 'pi'.")

        tr("start DailyServer.serve_day", minutes_budget=minutes_budget, policy=pol)

        # --- Daily breach snapshots (computed even if no capacity today) ---
        in_breach_start: Dict[str, int]    = {k: 0 for k in queues}
        new_breaches_today: Dict[str, int] = {k: 0 for k in queues}

        for kind, q in queues.items():
            if not q:
                continue
            dline = int(deadlines_min.get(kind, 10**9))
            for arr, pid in q:
                if arr >= day_start_min:
                    continue  # not yet waiting at day start
                wait_so_far = day_start_min - arr
                if wait_so_far > dline:          # strict '>' matches serve-time logic or should equality count?
                    in_breach_start[kind] += 1   # counts breaches as they happen
                    tr("breach at start", kind=kind, pid=pid, wait_so_far=wait_so_far, deadline=dline)
                if prev_day_start_min is not None:
                    breach_time = arr + dline
                    if prev_day_start_min < breach_time <= day_start_min:
                        new_breaches_today[kind] += 1 # breaches today regardless of served
                        tr("new_breach_today", kind=kind, pid=pid, breach_time=breach_time)

        # If no theatre time, still return the daily breach snapshot
        if minutes_budget <= 0:
            empty_counts = {k: 0 for k in queues}
            empty_lists  = {k: [] for k in queues}
            return (minutes_budget, empty_counts, empty_counts,
                    empty_lists, empty_lists, [],
                    in_breach_start, new_breaches_today)

        served_today: Dict[str, int]        = {k: 0 for k in queues}
        breached_today: Dict[str, int]      = {k: 0 for k in queues}
        waits_today: Dict[str, List[int]]   = {k: [] for k in queues}
        excess_today: Dict[str, List[int]]  = {k: [] for k in queues}
        patient_records: List[Dict[str, float | int | str | bool]] = []

        weights = dict(priority_weights or {})
        mean_dur = {k: float(mean_case_minutes.get(k, 1.0)) for k in queues}
        initial_budget = float(minutes_budget)

        def weight_for(kind: str) -> float:
            w = float(weights.get(kind, 1.0))
            return w if w > 0 else 1.0

        # >>> Track absolute time explicitly
        current_time = int(day_start_min)
        day_end_abs = int(day_start_min + initial_budget)

        while minutes_budget > 0:
            tr("Building candidates list", current_time=current_time, minutes_left=round(minutes_budget,1))
            candidates: List[Tuple[float, int, str, Tuple[int, int]]] = []  # (idx, arrival, kind, (arr,pid))

            for kind, q in queues.items():
                if not q:
                    continue
                dline = deadlines_min.get(kind, 10**9)
                w = weight_for(kind)
                e_dur = mean_dur.get(kind, 1.0)
                for arr, pid in q:
                    if arr > current_time:
                        continue  # not yet arrived
                    wait_now = current_time - arr
                    slack = dline - wait_now
                    if pol == "pi":
                        denom = max(1.0, e_dur * w)
                        idx = slack / denom
                    else:
                        idx = slack / max(w, 1e-9)
                    candidates.append((idx, arr, kind, (arr, pid)))

            if candidates:
                idx_vals = [c[0] for c in candidates]
                min_idx = min(idx_vals)
                ties = sum(1 for v in idx_vals if abs(v - min_idx) < 1e-9)
                tr("Candidates summary", n=len(candidates), min_idx=round(min_idx, 3), ties_at_min=ties)

            if not candidates:
                # FAST-FORWARD TO NEXT ARRIVAL (if any) 
                next_arrival = None
                for q in queues.values():
                    if q:
                        arr0 = q[0][0]  # earliest in that queue
                        if arr0 > current_time:
                            next_arrival = arr0 if next_arrival is None else min(next_arrival, arr0)

                if (next_arrival is not None) and (next_arrival <= day_end_abs):
                    tr("Fast-forwarding to next arrival",
                       from_time=current_time, to_time=next_arrival,
                       within_operating_window=True)
                    current_time = int(next_arrival)
                    # note: minutes_budget unchanged (idle time does not consume theatre minutes)
                    continue

                tr("No candidates left, break day loop",
                   current_time=current_time, day_end_abs=day_end_abs)
                break

            # Pick the best candidate - next line, if tie, who arrived first
            candidates.sort(key=lambda t: (t[0], t[1]))   # idx asc, then earlier arrival on ties
            idx_sel, _arr_key, kind, (arr, pid) = candidates[0]
            tr("Next patient", kind=kind, pid=pid, idx=idx_sel)

            # surgery duration
            dur = float(duration_sampler[kind]())
            tr("Sampled duration", kind=kind, pid=pid, dur=round(dur,1), minutes_budget_left=round(minutes_budget,1))

            # End-of-day packer: try to fit a smaller case if the top one doesn't fit
            # Utilisation is low if FIFO by PI is used.
            if minutes_budget < dur:
                tr("Top candidate does not fit",
                   top_kind=kind, top_pid=pid, top_dur=round(dur,1),
                   minutes_left=round(minutes_budget,1))
                packed = False
                for try_idx, (idx2, _arr2, kind2, (arr2, pid2)) in enumerate(candidates[1:11], start=1):
                    dur2 = float(duration_sampler[kind2]())
                    tr("Packer try", attempt=try_idx, cand_kind=kind2, cand_pid=pid2,
                       cand_dur=round(dur2,1), minutes_left=round(minutes_budget,1))
                    if minutes_budget >= dur2:
                        kind, arr, pid, dur = kind2, arr2, pid2, dur2
                        packed = True
                        tr("Packer chose", kind=kind, pid=pid, dur=round(dur,1))
                        break
                if not packed:
                    tr("Packer failed: stopping for the day",
                       minutes_left=round(minutes_budget,1))
                    break

            dline = deadlines_min.get(kind, 10**9)

            # True service times & true wait
            service_start = int(current_time)
            service_end   = int(round(service_start + dur))
            wait_true     = service_start - arr
            slack_now     = dline - wait_true
            breached_now  = (wait_true > dline)
            breach_excess = max(0, wait_true - dline)

            # Transparent logging
            wait_start  = day_start_min - arr
            slack_start = dline - wait_start
            tr("Pick", kind=kind, pid=pid, idx_raw=round(idx_sel,3),
               slack_start=int(slack_start), mean_dur=mean_dur.get(kind,1.0),
               weight=weight_for(kind), dur_sampled=round(dur,1))
            tr("Serving patient", kind=kind, pid=pid, wait=wait_true,
               dur=round(dur,1), slack_now=round(slack_now,1), deadline=dline)

            # Serve: consume theatre minutes and advance absolute time
            queues[kind].remove((arr, pid))
            minutes_budget -= dur
            current_time = service_end

            served_today[kind] += 1
            waits_today[kind].append(int(wait_true))
            if breached_now:
                breached_today[kind] += 1
                excess_today[kind].append(int(breach_excess))

            # Day progress summary (util uses only used minutes, not idle)
            used = initial_budget - minutes_budget
            util = 0.0 if initial_budget <= 0 else used / initial_budget
            tr("Day summary",
               served=sum(served_today.values()),
               served_breached=sum(breached_today.values()),
               minutes_left=round(minutes_budget,1),
               used=round(used,1),
               util=f"{util:.1%}")

            # Patient record
            patient_records.append({
                "id": pid,
                "kind": kind,
                "arrival": int(arr),
                "service_start": service_start,
                "service_end": service_end,
                "wait": int(wait_true),
                "duration": float(dur),
                "deadline": int(dline),
                "breached": bool(breached_now),                   # true when wait >= deadline
                "excess": int(breach_excess),                     # 0 if below deadline
                "policy": pol,
                "breach_time": int(arr + dline),
            })

        return (minutes_budget, served_today, breached_today,
                waits_today, excess_today, patient_records,
                in_breach_start, new_breaches_today)


# # Sampling arrivals and surgical duration
# 
# Using sim-tools

# In[ ]:


class DurationSampler:
    """
    Holds per-kind duration samplers using Config.duration_params.
    Uses Normal(mean, sigma, minimum, random_seed) from sim-tools
    """
    def __init__(self, cfg, rng_seed: int | None, trace=lambda *a, **k: None):
        self.cfg = cfg
        self.trace = trace
        self.rng_seed = 0 if rng_seed is None else int(rng_seed)
        self._samplers: Dict[str, Callable[[], float]] = self._build()

    def sample(self, kind: str) -> float:
        try:
            return float(self._samplers[kind]())
        except KeyError:
            raise KeyError(f"DurationSampler: unknown kind '{kind}'. "
                           f"Known kinds: {sorted(self._samplers.keys())}") from None

    def mean_minutes(self) -> Dict[str, float]:
        return self.cfg.get_mean_case_minutes()

    def as_dict(self) -> Dict[str, Callable[[], float]]:
        """Optional adapter if your server wants {kind: () -> float}."""
        return self._samplers

    def _build(self) -> Dict[str, Callable[[], float]]:
        dur_model = self.cfg.get_duration_model()   # {kind: {mean, sd, min}}
        kinds = sorted(dur_model.keys())            # stable order across runs
        ss = np.random.SeedSequence(self.rng_seed)
        seeds = ss.spawn(len(kinds))

        samplers: Dict[str, Callable[[], float]] = {}
        for kind, seed in zip(kinds, seeds):
            p = dur_model[kind]
            m, sd, mn = float(p["mean"]), float(p["sd"]), float(p.get("min", 0.0))
            # sim-tools with independent per-kind stream
            dist = Normal(mean=m, sigma=sd, minimum=mn, random_seed=seed)
            # late-binding safe via default arg
            samplers[kind] = (lambda d=dist: float(d.sample()))
            self.trace("Duration sampler built",
                       kind=kind, mean=m, sd=sd, minimum=mn, seed_entropy=seed.entropy)

        self.trace("DurationSampler ready",
                   base_seed=self.rng_seed, kinds=kinds, n=len(kinds))
        return samplers


# In[ ]:


class ArrivalProcess:
    """
    Pre-generates sorted absolute arrival times per kind over a horizon,
    using Exponential(mean_iat_minutes).
    """
    def __init__(self, cfg, horizon_days: int, rng_seed: Optional[int], trace=lambda *a, **k: None):
        self.cfg = cfg
        self.trace = trace
        self.horizon_days = int(horizon_days)
        self.rng_seed = 0 if rng_seed is None else int(rng_seed)

        self.times_by_kind: Dict[str, List[int]] = self._build()
        self.next_idx: Dict[str, int] = {k: 0 for k in self.times_by_kind}

        self.trace("ArrivalProcess ready",
                   base_seed=self.rng_seed,
                   horizon_min=self.horizon_days * 1440,
                   kinds=sorted(self.times_by_kind.keys()))

    # -------- public API --------

    def pop_arrivals_up_to(self, t_min: int) -> Dict[str, List[int]]:
        """Consume and return all arrivals with time <= t_min."""
        out = {k: [] for k in self.times_by_kind}
        for k, times in self.times_by_kind.items():
            i = self.next_idx[k]
            n = len(times)
            while i < n and times[i] <= t_min:
                out[k].append(times[i])
                i += 1
            self.next_idx[k] = i
        return out

    def remaining(self, kind: str) -> int:
        """How many arrivals remain for a kind (not yet popped)."""
        return len(self.times_by_kind[kind]) - self.next_idx[kind]

    def peek_next(self, kind: str) -> Optional[int]:
        """Next unconsumed arrival time for a kind, else None."""
        i = self.next_idx[kind]
        times = self.times_by_kind[kind]
        return times[i] if i < len(times) else None

    def reset(self) -> None:
        """Reset consumption indices (does not rebuild or reseed)."""
        self.next_idx = {k: 0 for k in self.times_by_kind}

    def __len__(self) -> int:
        """Total number of pre-generated arrivals (all kinds)."""
        return sum(len(v) for v in self.times_by_kind.values())

    # -------- internal --------

    def _build(self) -> Dict[str, List[int]]:
        horizon_min = self.horizon_days * 1440
        mean_iats = self.cfg.get_iat_mean_per_day()  # {kind: float minutes}
        kinds = sorted(mean_iats.keys())              # stable ordering

        ss = np.random.SeedSequence(self.rng_seed)
        seeds = ss.spawn(len(kinds))

        times: Dict[str, List[int]] = {k: [] for k in kinds}
        for kind, seed in zip(kinds, seeds):
            mean_iat = float(mean_iats[kind])
            if mean_iat <= 0:
                self.trace("ArrivalProcess skip kind (non-positive iat)", kind=kind, mean_iat=mean_iat)
                continue

            dist = Exponential(mean=mean_iat, random_seed=seed)
            t = 0.0
            cnt = 0
            # cumulative sum of exponentials => naturally sorted; no need to sort afterwards
            while t < horizon_min:
                t += float(dist.sample())
                if t >= horizon_min:
                    break
                times[kind].append(int(t))
                cnt += 1

            self.trace("Arrival stream built",
                       kind=kind, mean_iat=mean_iat, horizon_min=horizon_min,
                       n=cnt, seed_entropy=seed.entropy)

        return times


# ## Run model function
# 
# Run using either EDD or PI 
# The priority index (PI) is used in scheduling theory:
# 
# PI = slack / (mean duration * weight)
# 
# slack is the time to deadline
# mean duration is the **expected** duration, not the sampled duration
# 
# A smaller PI implies a more urgent job, pushed down by low slack (at or near deadline), long expected processing time, or a high weight.
# 
# Shorter jobs raise the index (reduce the priority) and the weight multiplies that effect. 
# 
# For surgery, it may be that favouring shorter surgery will increase throughput, but hips take longer in general, and are operationally more urgent (tariffs).
# 
# Prioritising longer surgery an introduce:  
#     * 1. A long idle tail at end of day
#     * FIX: try next candidate to find one that fits (end of day 'packer')  - done, check next ten in queue.
# 
# 

# In[ ]:


class Simulation:
    """
    A single replication given a Config - no global state.
    External dependencies (must be provided by caller/module):
      - ArrivalProcess(cfg, horizon_days, rng_seed, trace)
      - DurationSampler(cfg, rng_seed, trace) with .as_dict() -> {kind: () -> float} and .mean_minutes() -> {kind: float}
      - DailyServer(trace) with .serve_day(...)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.trace = cfg.trace  # injected tracer: callable(msg: str, **kwargs)
        self._validated = False #so can run validation() on first rep only

    def _minutes_capacity_for_day(self, sessions_by_dow: Dict[int, float], session_minutes, day: int) -> float:
        """Compute theatre minutes available for a given day index."""
        dow = day % 7
        sessions_today = float(sessions_by_dow.get(dow, 0.0))
        m_per_session = float(session_minutes(day)) if callable(session_minutes) else float(session_minutes)
        return sessions_today * m_per_session

    def single_run(self, run_id: int = 1) -> Dict[str, Any]:
        """
        One replication with policy = cfg.service_policy ("edd" or "pi").

        Outputs:
          - "patients": patient-level records for BOTH served and unserved patients.
          - "kpis": run-level aggregates (post-warmup where noted), with detailed comments below.
        """
        cfg = self.cfg
        tr = self.trace

        # --- Setup & config normalization ---
        tr("Begin single_run", run_id=run_id)
        if not self._validated:
            self.cfg.validate()
            self._validated = True #after first run

        warmup_cut_min  = cfg.get_warmup_cut_min()
        horizon_days    = cfg.get_horizon_days()
        end_time_min    = cfg.get_horizon_end_min()
        tr("Initial constants",
           warmup_days=cfg.get_warmup_days(),
           warmup_cut_min=warmup_cut_min,
           horizon_days=horizon_days,
           horizon_end_min=end_time_min)

        base_seed        = cfg.get_base_seed() + int(run_id)
        sessions_by_dow  = cfg.get_sessions_by_dow()
        session_minutes  = cfg.get_session_minutes()
        deadlines_min    = cfg.get_breach_deadlines_minutes()
        policy           = cfg.get_policy()
        priority_weights = cfg.get_priority_weights()
        tr("Base seed", value=base_seed)
        tr("Policy & weights", policy=policy, weights=priority_weights)

        # --- Stochastic inputs ---
        arrivals = ArrivalProcess(cfg, horizon_days, rng_seed=base_seed + 10_000, trace=tr)
        dur      = DurationSampler(cfg, rng_seed=base_seed + 20_000, trace=tr)
        duration_sampler  = dur.as_dict()     # {kind: () -> float}
        mean_case_minutes = dur.mean_minutes()# {kind: float}

        # --- State ---
        patient_id_counter: itertools.count = itertools.count(1)
        queues: Dict[str, Deque[Tuple[int, int]]] = {k: deque() for k in arrivals.times_by_kind.keys()}

        # --- Aggregates (post-warmup, by kind unless stated otherwise) ---
        served_counts         = {k: 0 for k in queues}  # served patient counts by kind
        breached_counts       = {k: 0 for k in queues}  # served-in-breach counts by kind
        waits_all             = {k: [] for k in queues} # waits (minutes) for served patients by kind
        excess_all            = {k: [] for k in queues} # excess over deadline (minutes) for served, by kind

        in_breach_start_tot   = {k: 0 for k in queues}  # daily snapshot: already-breached at start-of-day, summed over post-warmup days
        new_breaches_tot      = {k: 0 for k in queues}  # daily snapshot: newly crossed breach threshold during the day, summed post-warmup
        in_breach_end_tot     = {k: 0 for k in queues}  # daily snapshot: still-waiting & in-breach at end-of-day, summed post-warmup

        # Utilisation (post-warmup only)
        util_minutes_used_total      = 0.0  # sum of minutes actually used across post-warmup days
        util_minutes_capacity_total  = 0.0  # sum of available capacity minutes across post-warmup days
        utilisation_by_day_post_warmup: List[dict] = []  # [{day, minutes_used, minutes_capacity, utilisation}]

        # Patient-level records
        patient_records: List[Dict[str, Any]] = []

        # Sanity trace per kind
        for k in queues:
            tr("Param check",
               kind=k,
               weight=priority_weights.get(k, 1.0),
               mean_case_minutes_runtime=mean_case_minutes.get(k),
               deadline_min=deadlines_min.get(k))

        server = DailyServer(trace=tr)

        # --- Main day loop ---
        for day in range(horizon_days):
            day_start_min = day * 1440
            prev_day_start_min = (day - 1) * 1440 if day > 0 else None
            tr("==== NEW DAY ====", day=day, start_min=day_start_min, prev_day_start_min=prev_day_start_min)

            # Capacity for the day
            minutes_budget_today = self._minutes_capacity_for_day(sessions_by_dow, session_minutes, day)
            tr("Daily capacity minutes", day=day, minutes_budget_today=minutes_budget_today)

            # Arrivals: include up to the last minute we could still be operating today
            arrivals_cutoff = day_start_min + int(minutes_budget_today)
            new_arrs = arrivals.pop_arrivals_up_to(arrivals_cutoff)
            new_count = {k: len(v) for k, v in new_arrs.items()}
            tr("Arrivals pulled", day=day, cutoff=arrivals_cutoff, counts=new_count)

            # Push arrivals into the queues
            for kind, arr_list in new_arrs.items():
                for arr in arr_list:
                    pid = next(patient_id_counter)
                    queues[kind].append((arr, pid))

            # Serve once per day
            (
                minutes_left,            # minutes remaining unused
                s_today,                 # served counts by kind
                b_today,                 # served-in-breach counts by kind
                w_today,                 # list of waits for served by kind
                x_today,                 # list of excess waits for served by kind
                day_patients,            # patient-level records for those served today
                in_breach_start_day,     # snapshot counts at day start (already breached), by kind
                new_breaches_day,        # snapshot counts of newly breached during the day, by kind
            ) = server.serve_day(
                queues=queues,
                minutes_budget=minutes_budget_today,
                day_start_min=day_start_min,
                policy=policy,
                deadlines_min=deadlines_min,
                duration_sampler=duration_sampler,      # dict[str, ()->float]
                mean_case_minutes=mean_case_minutes,
                priority_weights=priority_weights,
                prev_day_start_min=prev_day_start_min,
            )

            # Utilisation for today (independent of warmup; aggregated post-warmup below)
            used_today = max(0.0, float(minutes_budget_today) - float(minutes_left))
            tr("Utilisation today",
               day=day,
               minutes_used=used_today,
               minutes_capacity=float(minutes_budget_today),
               minutes_left=float(minutes_left))

            # Tag warm-up on served patients & record
            for rec in day_patients:
                rec["in_warmup"] = (rec["service_start"] is not None
                                    and rec["service_start"] < warmup_cut_min)
            patient_records.extend(day_patients)
            tr("Served today (by kind)", day=day, served=s_today, served_in_breach=b_today)

            # End-of-day breached backlog snapshot (by kind), after serving
            in_breach_end_day = {k: 0 for k in queues}
            day_end_min = day_start_min + 1440
            for k, q in queues.items():
                dline = deadlines_min[k]
                for arr, _pid in q:
                    if (day_end_min - arr) > dline:
                        in_breach_end_day[k] += 1
            tr("End-of-day breached backlog", day=day, in_breach_end_day=in_breach_end_day)

            # Accumulate KPIs after warm-up only
            if day_start_min >= warmup_cut_min:
                for k in s_today: served_counts[k]   += s_today[k]
                for k in b_today: breached_counts[k] += b_today[k]
                for k in w_today: waits_all[k]       += w_today[k]
                for k in x_today: excess_all[k]      += x_today[k]
                for k in in_breach_start_day: in_breach_start_tot[k] += in_breach_start_day[k]
                for k in new_breaches_day:    new_breaches_tot[k]    += new_breaches_day[k]
                for k in in_breach_end_day:   in_breach_end_tot[k]    += in_breach_end_day[k]

                util_minutes_used_total     += used_today
                util_minutes_capacity_total += float(minutes_budget_today)
                utilisation_by_day_post_warmup.append({
                    "day": day,
                    "minutes_used": used_today,
                    "minutes_capacity": float(minutes_budget_today),
                    "utilisation": (used_today / float(minutes_budget_today)) if minutes_budget_today > 0 else 0.0,
                })
                tr("Post-warmup accumulation",
                   day=day,
                   added_served=s_today,
                   added_breached=b_today,
                   added_in_breach_start=in_breach_start_day,
                   added_new_breaches=new_breaches_day,
                   added_in_breach_end=in_breach_end_day,
                   util_used_total=util_minutes_used_total,
                   util_cap_total=util_minutes_capacity_total)

        # --- End-of-horizon snapshot for remaining (unserved) patients ---
        waiting_count_final = sum(len(q) for q in queues.values())
        waiting_minutes_final = sum(max(0, end_time_min - arr) for q in queues.values() for arr, _ in q)

        breached_waiting_final = 0
        breached_waiting_minutes_final = 0
        for kind, q in queues.items():
            deadline_min = deadlines_min[kind]
            for arrival_min, pid in q:
                wait   = end_time_min - arrival_min
                excess = wait - deadline_min
                breached_now = (wait > deadline_min) #actual breaches
                if breached_now:
                    breached_waiting_final += 1
                    breached_waiting_minutes_final += int(max(0, excess))
                #breach_time_val = (arrival_min + deadline_min) if breached_now else None
                patient_records.append({
                    "id": pid,
                    "kind": kind,
                    "arrival": int(arrival_min),
                    "deadline": int(deadline_min),
                    "service_start": None,
                    "service_end": None,
                    "wait": int(wait),
                    "excess": int(max(0, excess)),
                    "breached": bool(breached_now),
                    #"breach_time": int(breach_time_val) if breach_time_val is not None else None,
                    "in_warmup": arrival_min < warmup_cut_min,
                    "policy": policy,
                    "breach_time": int(arrival_min + deadline_min),
                })
        tr("End-of-horizon unserved snapshot",
           waiting_count_final=int(waiting_count_final),
           waiting_minutes_final=int(waiting_minutes_final),
           breached_waiting_final=int(breached_waiting_final),
           breached_waiting_minutes_final=int(breached_waiting_minutes_final))

        # --- Breach incidence from timestamps (post-warmup, within horizon) ---
        breach_incidence_by_kind = defaultdict(int)

        for rec in patient_records:
            bt = rec.get("breach_time")
            if bt is None or not (warmup_cut_min <= bt < end_time_min):
                continue

            ss = rec.get("service_start")
            # Count only if the patient was *still waiting* at bt.
            # Served: breach iff service_start > bt (strictly later than deadline).
            # Unserved: breach iff end_of_horizon is after bt (already ensured by bt < end_time_min).
            if (ss is None) or (ss > bt):
                breach_incidence_by_kind[rec["kind"]] += 1

        breach_incidence_total = int(sum(breach_incidence_by_kind.values()))   

        # Utilisation summary (post-warmup days only)
        utilisation_rate_post_warmup = (
            float(util_minutes_used_total) / util_minutes_capacity_total
            if util_minutes_capacity_total > 0 else 0.0
        )
        tr("Utilisation summary (post-warmup)",
           minutes_used_total=util_minutes_used_total,
           minutes_capacity_total=util_minutes_capacity_total,
           utilisation_rate=utilisation_rate_post_warmup)

        # --- KPIs (with explanations) ---

        # Convenience totals
        served_total              = int(sum(served_counts.values()))
        breached_served_total     = int(sum(breached_counts.values()))                 # served while already past deadline
        #breach_incidence_total    = int(sum(new_breaches_tot.values()))                # true events: first time a patient crosses deadline

        #  by-kind derived numbers
        served_within_deadline_by_kind = {k: int(served_counts[k] - breached_counts[k]) for k in served_counts}
        served_within_deadline_total   = int(sum(served_within_deadline_by_kind.values()))

        #  simple rates (guarded against divide-by-zero)
        pct_within_deadline_overall = (served_within_deadline_total / served_total) if served_total > 0 else 0.0
        pct_within_deadline_by_kind = {
            k: (served_within_deadline_by_kind[k] / served_counts[k]) if served_counts[k] > 0 else 0.0
            for k in served_counts
        }

        kpis = {
            "policy": policy,  # Service policy used (edd or pi)

            # -----------------------------
            # FLOWS: Served patients (post-warmup; patient counts)
            # -----------------------------
            "served_by_kind": served_counts,                     # Patients served, by kind, after warm-up.
            "served_total": served_total,                        # Total served after warm-up.

            # Performance view of served patients relative to the SLA at service start:
            "breached_served_by_kind": breached_counts,          # Of the served, how many were already past deadline at service start (post-warmup).
            "breached_served_total": breached_served_total,      # Total served-while-in-breach (post-warmup).

            # Compliance view for served patients:
            "served_within_deadline_by_kind": served_within_deadline_by_kind,  # Served at/under deadline, by kind (post-warmup).
            "served_within_deadline_total": served_within_deadline_total,      # Total served at/under deadline (post-warmup).
            "pct_within_deadline_by_kind": pct_within_deadline_by_kind,        # Proportion of served within deadline, by kind (0..1).
            "pct_within_deadline_overall": pct_within_deadline_overall,        # Overall proportion within deadline (0..1).

            # -----------------------------
            # EVENTS: Breach incidence (official breach metric; post-warmup)
            # -----------------------------
            # These are *true events* counted once when a patient crosses the SLA (arrival_time + deadline).
            "breach_incidence_by_kind": dict(breach_incidence_by_kind),        # Number of first-time breaches during the day, summed over post-warmup days.
            "breach_incidence_total": breach_incidence_total,    # Total breach events across kinds (post-warmup).

            # Back-compat aliases so existing downstream code that expects 'breached_*' keeps working,
            # but now points to the event-based definition:
            "breached_by_kind": dict(breach_incidence_by_kind),                # ALIAS to incidence (preferred: use 'breach_incidence_by_kind').
            "breached_total": breach_incidence_total,            # ALIAS to incidence (preferred: use 'breach_incidence_total').

            # -----------------------------
            # STOCKS: End-of-horizon waiting (served + unserved context)
            # -----------------------------
            "waiting_count_final": int(waiting_count_final),     # Number of patients still waiting at the end of the horizon.
            "waiting_minutes_final": int(waiting_minutes_final), # Sum of waits (end_time - arrival) for those still waiting.

            # Among the unserved at horizon end, breach status (using >= deadline):
            "breached_waiting_final": int(breached_waiting_final),                   # Count of still-waiting who are in breach at horizon end.
            "breached_waiting_minutes_final": int(breached_waiting_minutes_final),   # Sum of (wait - deadline, floored at 0) for those in breach.

            # -----------------------------
            # EXPOSURE-DAY SNAPSHOTS (post-warmup sums)
            # -----------------------------
            # These are *exposures*, not people: each day contributes counts; a single patient can contribute on multiple days.
            "in_breach_start_total_by_kind": in_breach_start_tot,  # At day start (08:00), already-in-breach; summed over post-warmup days.
            "in_breach_end_total_by_kind": in_breach_end_tot,      # At day end (24:00), still waiting & in-breach; summed post-warmup.
            # Kept for completeness/reference; these same counts feed 'breach_incidence_*' totals above:
            "new_breaches_total_by_kind": new_breaches_tot,        # Exposure snapshots (not equal to incidence totals; last day events are not included)

            # -----------------------------
            # WAIT DISTRIBUTIONS (served; post-warmup samples)
            # -----------------------------
            # Raw samples for downstream summaries/plots; minutes from arrival to service start.
            "waits_by_kind_minutes": waits_all,                   # List[int] wait times for served patients, by kind (post-warmup).
            "excess_by_kind_minutes": excess_all,                 # List[int] minutes over deadline at service start (0 if on time), by kind (post-warmup).

            # -----------------------------
            # UTILISATION (post-warmup only)
            # -----------------------------
            "util_minutes_used_total_post_warmup": util_minutes_used_total,          # Sum of minutes actually used across post-warmup days.
            "util_minutes_capacity_total_post_warmup": util_minutes_capacity_total,  # Sum of available minutes across post-warmup days.
            "utilisation_rate_post_warmup": utilisation_rate_post_warmup,            # Used / Capacity over post-warmup days (0..1).
            "utilisation_by_day_post_warmup": utilisation_by_day_post_warmup,        # Per-day series for plotting/diagnostics.
        }


        return {"run_id": run_id, "seed_used": base_seed, "kpis": kpis, "patients": patient_records}

    # Convenience: batch runner that mirrors your loop
    def run_many(self, n_reps: int) -> Tuple[List[Dict[str, Any]], Dict[int, List[dict]]]:
        rows: List[Dict[str, Any]] = []
        results_dict: Dict[int, List[dict]] = {}
        for r in range(1, int(n_reps) + 1):
            res = self.single_run(run_id=r)
            rows.append({"rep": r, **res["kpis"]})
            results_dict[r] = res["patients"]
        return rows, results_dict


# # Get a clean copy of config ready for scenarios

# In[ ]:


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


# ## tests

# In[ ]:


# Ensure base cfg is not mutated by overrides
_base = copy.deepcopy(cfg)
cfg_test = apply_overrides(cfg, {"priority_weights": {"hip": 9.99}})
assert cfg.priority_weights["hip"] == _base.priority_weights["hip"], "Base cfg was mutated!"
print("passed")


# In[ ]:


# 1) Instance ownership (i.e. no shared dicts)
c1, c2 = Config(), Config()
c1.priority_weights["hip"] = 9.99
assert c2.priority_weights["hip"] != 9.99

# 2) Stable seeds
cfg = Config()
s1 = cfg.seed_for(1, stream="arrivals")
s2 = cfg.seed_for(1, stream="arrivals")
assert s1 == s2  # should always hold across runs/processes

# 3) Weekly capacity > 0 with defaults
assert cfg.get_total_weekly_capacity_minutes() > 0
print("all tests passed")


# # Example run scenarios

# In[ ]:


# --- Base config ---
cfg = Config()
cfg.horizon_days = 350
cfg.warmup_days  = 35
cfg.session_minutes = 240
cfg.trauma_sessions_by_dow = {0:4,1:2,2:4,3:2,4:4,5:2,6:2}
cfg.iat_mean_per_day = {"hip":360.0, "shoulder":1440.0, "wrist":480.0, "ankle":720.0}
cfg.duration_params = {
    "hip": {"mean":90.0, "sd":25.0, "min":30.0},
    "shoulder":{"mean":75.0, "sd":22.5, "min":20.0},
    "wrist":{"mean":60.0, "sd":18.0, "min":15.0},
    "ankle":{"mean":55.0, "sd":27.0, "min":20.0},
}
cfg.breach_deadline_minutes = {"hip":36*60, "shoulder":14*24*60, "wrist":14*24*60, "ankle":14*24*60}
cfg.service_policy = "pi"
cfg.priority_weights = {"hip":2.0,"shoulder":1.0,"wrist":1.0,"ankle":1.0}
cfg.base_seed = 1984
cfg.TRACE = False

# --- Single run ---
res = run_single(cfg, run_id=1)
print("KPIs:", list(res["kpis"].keys()))

# # --- Reps ---
# rows, results_dict = run_reps(cfg, n_reps=30)
# df, desc, summary = summarize_reps(rows)
# if df is not None:
#     print("Replication summary:"); print(desc)
#     print("\nKPI means ±95% CI:"); print(summary)

# --- Scenarios ---
scenarios = {
  "baseline_pi": {},
  "more_hips_priority": {"priority_weights": {"hip": 3.0}},
  "edd_policy": {"service_policy": "edd"},
  "more_capacity": {"trauma_sessions_by_dow": {0:5,1:2,2:4,3:2,4:4,5:2,6:2}},
}

kpi_rows, patients_by_scen = run_scenarios(cfg, scenarios, n_reps=300, attach_patients_last_only=False)
df_scen = scenarios_to_df(kpi_rows)
if df_scen is not None:
    print("\nScenario KPI head:")
    print(df_scen.head())


# ## Tests/validate scenarios

# In[ ]:


# rebuild patients df across scenarios/reps
def patients_to_df(pby):
    frames=[]
    for scen, rep_map in pby.items():
        for rep, plist in rep_map.items():
            if not plist: continue
            d=pd.DataFrame(plist)
            if d.empty: continue
            d["scenario"]=scen; d["rep"]=int(rep)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

patients_df = patients_to_df(patients_by_scen).drop_duplicates(subset=["scenario","rep","id"])

warmup_cut_min = cfg.get_warmup_cut_min()
end_time_min   = cfg.get_horizon_end_min()

# ground-truth incidence from timestamps: warm-up window & still-waiting at bt
pt_breach = (patients_df
    .query("@warmup_cut_min <= breach_time < @end_time_min")
    .assign(_still_waiting=lambda d: d["service_start"].isna() | (d["service_start"] > d["breach_time"]))
    .query("_still_waiting")
    .groupby(["scenario","rep"], observed=True)
    .size().rename("breach_total_pt").reset_index())

df_kpis = pd.json_normalize(kpi_rows)
cmp = (df_kpis[["scenario","rep","breach_incidence_total"]]
       .merge(pt_breach, on=["scenario","rep"], how="left").fillna(0))

# these should now match exactly
assert (cmp["breach_incidence_total"] - cmp["breach_total_pt"]).abs().max() == 0

# served totals should already match
pt_served = (patients_df
    .query("service_start.notna() and service_start >= @warmup_cut_min")
    .groupby(["scenario","rep"], observed=True)
    .size().rename("served_total_pt").reset_index())

cmp2 = (df_kpis[["scenario","rep","served_total"]]
        .merge(pt_served, on=["scenario","rep"], how="left").fillna(0))
assert (cmp2["served_total"] - cmp2["served_total_pt"]).abs().max() == 0

print("passed")


# In[ ]:


cfg.validate()
print("config ok")

# confirm each scenario actually changed what you think
for name, ov in scenarios.items():
    c = apply_overrides(cfg, ov)
    print(f"\n{name}")
    print(" policy:", c.get_policy(), "weights:", c.get_priority_weights()["hip"],
          "Mon sessions:", c.get_sessions_by_dow().get(0))


# In[ ]:


df_kpis = pd.json_normalize(kpi_rows)
counts = (df_kpis.groupby("scenario")["rep"]
                  .nunique()
                  .rename("#reps"))
print("\nrep counts per scenario:\n", counts)


# In[ ]:


def summarize_kpis(kpi_rows):
    df = pd.json_normalize(kpi_rows)
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ("rep",)]
    df_num = df[["scenario","rep"] + num_cols]

    g = df_num.groupby("scenario", as_index=False)
    mean = g[num_cols].mean()
    sd   = g[num_cols].std(ddof=1).rename(columns={c:f"{c}_sd" for c in num_cols})
    n    = g.size().rename(columns={"size":"n"})
    out  = mean.merge(sd, on="scenario").merge(n, on="scenario")
    for c in num_cols:
        out[f"{c}_ci95"] = 1.96 * (out[f"{c}_sd"] / np.sqrt(out["n"]))
    return df_num, out

df_kpis_num, kpi_summary = summarize_kpis(kpi_rows)
print("\nKPI summary (mean ±95% CI) — selected columns:")
cols = [c for c in kpi_summary.columns if c.startswith(("scenario",
    "served_total","breach_incidence_total","pct_within_deadline_overall",
    "utilisation_rate_post_warmup")) or c=="n"]
print(kpi_summary[cols].round(3))


# In[ ]:


# prefer event-based breaches; fall back if needed
def pick_breach_colnames(df):
    if any(c.startswith("breach_incidence_by_kind.") for c in df.columns):
        pref = "breach_incidence_by_kind."
    elif any(c.startswith("breached_by_kind.") for c in df.columns):
        pref = "breached_by_kind."
    else:
        pref = "breached_served_by_kind."
    return [c for c in df.columns if c.startswith(pref)], pref

by_kind_cols, pref = pick_breach_colnames(df_kpis)
served_cols = [c for c in df_kpis.columns if c.startswith("served_by_kind.")]

by_kind = (df_kpis[["scenario"] + by_kind_cols + served_cols]
           .groupby("scenario", as_index=False).mean(numeric_only=True))

print("\nMean served_by_kind per scenario:")
print(by_kind[["scenario"] + served_cols].round(1))

print(f"\nMean {pref} per scenario:")
print(by_kind[["scenario"] + by_kind_cols].round(1))


# In[ ]:


# check that different reps used different seeds (proxy: served_total varies)
vt = df_kpis_num.pivot_table(index="rep", columns="scenario", values="served_total")
print("\nserved_total variance across reps (shouldn’t be all zeros):")
print(vt.var().round(2))


# In[ ]:


# KPI summary across reps (per scenario)


# In[ ]:


def summarize_kpis(kpi_rows):
    df = pd.json_normalize(kpi_rows)

    # numeric KPI columns (exclude the replication index)
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ("rep",)]

    # keep scenario/rep + numeric KPIs
    df_num = df[["scenario", "rep"] + num_cols].copy()

    # group by scenario (keep 'scenario' as a column)
    mean = df_num.groupby("scenario", as_index=False)[num_cols].mean()
    sd   = df_num.groupby("scenario", as_index=False)[num_cols].std(ddof=1)
    n    = df_num.groupby("scenario", as_index=False).size().rename(columns={"size":"n"})

    # add suffix to SD columns
    sd = sd.rename(columns={c: f"{c}_sd" for c in sd.columns if c != "scenario"})

    # merge on 'scenario'
    out = mean.merge(sd, on="scenario").merge(n, on="scenario")

    # 95% CI half-widths
    for c in num_cols:
        out[f"{c}_ci95"] = 1.96 * (out[f"{c}_sd"] / np.sqrt(out["n"]))

    return df, out


df_kpis, kpi_summary = summarize_kpis(kpi_rows)
print("KPI summary (mean ±95% CI):")
print(kpi_summary)



# In[ ]:


## Patient level table across reps (per scenario)


# In[ ]:


def patients_to_df(patients_by_scen):
    frames = []
    for scen, rep_map in patients_by_scen.items():
        items = rep_map.items() if isinstance(rep_map, dict) else [(1, rep_map)]
        for rep, plist in items:
            if not plist:
                continue
            d = pd.DataFrame(plist)
            if d.empty:
                continue
            d["scenario"] = scen
            d["rep"] = int(rep)
            frames.append(d)

    # keep only frames that have at least one non-NA value
    frames = [f for f in frames if (not f.empty) and f.notna().any().any()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

patients_df = patients_to_df(patients_by_scen)
print("Patient rows:", len(patients_df))


# # Plots

# ## KPI plots

# ### Helpers - reshape and CIs

# In[ ]:


def _is_scalar_series(s: pd.Series) -> bool:
    # treat numbers/bools as scalar; exclude lists/dicts/arrays
    return (pd.api.types.is_number(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)) \
           and not s.apply(lambda x: isinstance(x, (dict, list, tuple, np.ndarray)) ).any()

def melt_kpis(
    df_scen: pd.DataFrame,
    kpis: list[str] | None = None,
    *,
    id_cols: tuple[str, ...] = ("scenario", "rep"),
    prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] | None = ("utilisation_by_day_",),  # avoid per-day arrays by default
    scenario_order: list[str] | None = None,
    value_name: str = "value",
    dropna: bool = True,
    coerce_numeric: bool = False,
) -> pd.DataFrame:
    """
    Wide -> long for KPI plotting across reps.

    - If `kpis` is None, auto-select scalar numeric columns (excluding id_cols and excluded prefixes).
    - If `prefixes` is provided, include columns starting with any prefix (union with `kpis`).
    - `exclude_prefixes` skips unwanted wide columns (e.g., per-day arrays).
    """
    df = df_scen.copy()
    present_ids = [c for c in id_cols if c in df.columns]

    # auto-pick KPI columns that are scalar per cell
    if kpis is None:
        candidate_cols = [c for c in df.columns if c not in present_ids]
        if exclude_prefixes:
            candidate_cols = [c for c in candidate_cols if not any(c.startswith(p) for p in exclude_prefixes)]
        # keep only scalar numeric/bool columns
        scalar_cols = [c for c in candidate_cols if _is_scalar_series(df[c])]
        kpis = scalar_cols

    # add any requested prefixes
    if prefixes:
        pref_cols = [c for c in df.columns for p in prefixes if c.startswith(p)]
        kpis = sorted(set(kpis).union(pref_cols))

    # validate
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
    z: float = 1.96,  # for large n; swap to t-multiplier if you prefer
    observed: bool = True,
) -> pd.DataFrame:
    agg = (df_long.groupby(list(group_cols), as_index=False)
                  .agg(mean=(val, "mean"), n=(val, "size"), sd=(val, "std")))
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["lo"] = agg["mean"] - z * agg["se"]
    agg["hi"] = agg["mean"] + z * agg["se"]
    return agg


# ## examples of kpis with error bars

# In[ ]:


# df_scen = scenarios_to_df(kpi_rows)
long = melt_kpis(
    df_scen,
    prefixes=("breach_incidence_total", "pct_within_deadline_overall",
              "waiting_count_final", "utilisation_rate_post_warmup")
)
summ = ci95(long)

# Plot a single KPI (e.g., breach incidence) by scenario
to_plot = summ[summ["kpi"].eq("breach_incidence_total")]
px.bar(to_plot, x="scenario", y="mean", error_y=to_plot["hi"]-to_plot["mean"],
       error_y_minus=to_plot["mean"]-to_plot["lo"],
       title="Breach incidence (mean ±95% CI)").show()

# Or facet multiple KPIs at once
px.bar(summ, x="scenario", y="mean", color="scenario", facet_col="kpi", facet_col_wrap=2,
       error_y=summ["hi"]-summ["mean"], error_y_minus=summ["mean"]-summ["lo"],
       title="KPIs by scenario (mean ±95% CI)").show()


# ## 1. Throughput and breaches: served_by_kind and breach_incidence_by_kind

# In[ ]:


AMB_KINDS = ("shoulder", "wrist", "ankle")

def make_grouped_long(df_scen: pd.DataFrame, amb_kinds=AMB_KINDS) -> pd.DataFrame:
    df = df_scen.copy()

    # Helper to fetch breach counts per kind (tries incidence first, then aliases)
    breach_prefixes = (
        "breach_incidence_by_kind.",      # preferred
        "breached_by_kind.",              # alias you may have set to incidence
        "breached_served_by_kind.",       # legacy served-while-in-breach
    )
    def get_breach_val(row, kind):
        for pref in breach_prefixes:
            key = f"{pref}{kind}"
            if key in row and pd.notna(row[key]):
                return row[key]
        return np.nan

    records = []
    for _, r in df.iterrows():
        # Served totals by group
        hip_served = r.get("served_by_kind.hip", np.nan)
        amb_served = sum(r.get(f"served_by_kind.{k}", 0) for k in amb_kinds)

        # Breach totals by group (using preferred metric available)
        hip_breach = get_breach_val(r, "hip")
        amb_breach = sum(get_breach_val(r, k) or 0 for k in amb_kinds)

        for group, served, breached in (
            ("hip", hip_served, hip_breach),
            ("ambulatory", amb_served, amb_breach),
        ):
            records.append({"scenario": r["scenario"], "rep": r["rep"], "group": group,
                            "kpi": "served_total", "value": served})
            records.append({"scenario": r["scenario"], "rep": r["rep"], "group": group,
                            "kpi": "breached_total", "value": breached})

    long = pd.DataFrame.from_records(records)
    return long


# ## Boxplot per scenario

# In[ ]:


long = make_grouped_long(df_scen)

fig = px.box(
    long, x="scenario", y="value", color="group",
    facet_col="kpi", facet_col_wrap=2, points="all",
    title="Throughput & Breaches by Group (per replication)"
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None)
fig.show()


# ## Means with 95%CI pre scenario and patient type

# In[ ]:


# Double check that data is internally consistent
# Which served_by_kind columns exist?
served_cols = [c for c in df_scen.columns if c.startswith("served_by_kind.")]
print("served_by_kind cols:", served_cols)

# Check that per-kind sums match served_total per (scenario, rep)
chk = df_scen[["scenario","rep","served_total"] + served_cols].copy()
chk["sum_kinds"] = chk[served_cols].sum(axis=1, numeric_only=True)
print((chk["served_total"] - chk["sum_kinds"]).describe())


# In[ ]:


AMB_KINDS = ("shoulder", "wrist", "ankle")
WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def patients_to_df(pby):
    frames=[]
    for scen, rep_map in pby.items():
        for rep, plist in rep_map.items():
            if not plist: continue
            d=pd.DataFrame(plist)
            if d.empty: continue
            d["scenario"]=scen; d["rep"]=int(rep)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def make_weekday_long(patients_by_scen, cfg) -> pd.DataFrame:
    """Return tidy per-(scenario,rep,weekday,group,kpi,value) for served & breach incidence."""
    pdf = patients_to_df(patients_by_scen).drop_duplicates(subset=["scenario","rep","id"]).copy()
    if pdf.empty:
        return pd.DataFrame(columns=["scenario","rep","weekday","weekday_name","group","kpi","value"])

    # ensure breach_time exists (belt-and-braces)
    if "breach_time" not in pdf.columns:
        pdf["breach_time"] = np.nan
    need_bt = pdf["breach_time"].isna() & pdf["breached"].astype("boolean", errors="ignore").fillna(False)
    pdf.loc[need_bt, "breach_time"] = pdf.loc[need_bt, "arrival"].astype("Int64") + pdf.loc[need_bt, "deadline"].astype("Int64")

    warm = cfg.get_warmup_cut_min()
    end  = cfg.get_horizon_end_min()

    pdf["group"] = np.where(pdf["kind"].eq("hip"), "hip", "ambulatory")

    recs = []

    # --- Served by weekday (post-warmup) ---
    served = pdf[(pdf["service_start"].notna()) & (pdf["service_start"] >= warm)].copy()
    served["weekday"] = ((served["service_start"] // 1440) % 7).astype(int)
    g_served = (served
                .groupby(["scenario","rep","group","weekday"], observed=True)
                .size().rename("value").reset_index())
    g_served["kpi"] = "served_total"
    recs.append(g_served)

    # --- Breach incidence by weekday (post-warmup, still waiting at breach time) ---
    breach = pdf[(pdf["breach_time"].notna()) & (pdf["breach_time"] >= warm) & (pdf["breach_time"] < end)].copy()
    # count only if still waiting at breach_time
    breach["_still_waiting"] = breach["service_start"].isna() | (breach["service_start"] > breach["breach_time"])
    breach = breach[breach["_still_waiting"]].copy()
    breach["weekday"] = ((breach["breach_time"] // 1440) % 7).astype(int)
    g_breach = (breach
                .groupby(["scenario","rep","group","weekday"], observed=True)
                .size().rename("value").reset_index())
    g_breach["kpi"] = "breached_total"
    recs.append(g_breach)

    long = pd.concat(recs, ignore_index=True)
    long["weekday_name"] = pd.Categorical(long["weekday"].map(dict(enumerate(WEEKDAYS))), categories=WEEKDAYS, ordered=True)
    return long

def ci95(df, val="value", z=1.96, group_cols=("scenario","group","kpi","weekday_name")):
    g = df.groupby(list(group_cols), as_index=False, observed=True)
    out = g.agg(mean=(val,"mean"), n=(val,"size"), sd=(val,"std"))
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["lo"] = out["mean"] - z*out["se"]
    out["hi"] = out["mean"] + z*out["se"]
    return out

# --- Build weekday summary
weekday_long = make_weekday_long(patients_by_scen, cfg)
agg = ci95(weekday_long)

# --- Plot: mean ±95% CI by weekday
fig = px.line(
    agg, x="weekday_name", y="mean", color="scenario",
    line_dash="group",  # hip vs ambulatory
    facet_col="kpi", facet_col_wrap=2,
    error_y=agg["hi"]-agg["mean"], error_y_minus=agg["mean"]-agg["lo"],
    markers=True,
    title="Served & Breach Incidence by Weekday (mean ±95% CI across reps)"
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_xaxes(title=None)
fig.update_yaxes(matches=None)
fig.show()


# In[ ]:


kpi = "breached_total"  # or "served_total"

heat_df = (agg[agg["kpi"].eq(kpi)]
           .pivot_table(index=["scenario","group"],
                        columns="weekday_name",
                        values="mean",
                        observed=False)
           .reindex(columns=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]))

# Make single y labels like "baseline_pi • ambulatory"
ylabels = [f"{s} • {g}" for s, g in heat_df.index]

fig = px.imshow(
    heat_df.values,
    x=heat_df.columns,
    y=ylabels,
    labels=dict(x="Weekday", y="Scenario • Group", color=f"mean {kpi}"),
    aspect="auto",
    text_auto=True,
)
fig.update_layout(title=f"{kpi}: mean by weekday (across reps)")
fig.update_coloraxes(colorscale="Viridis", cmin=0, cmax=80)
fig.show()



# In[ ]:


import numpy as np
import plotly.express as px

WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

for kpi in ["served_total", "breached_total"]:
    dat = agg[agg["kpi"].eq(kpi)].copy()

    # lengths for error bars
    dat["errp"] = (dat["hi"] - dat["mean"]).fillna(0)
    dat["errm"] = (dat["mean"] - dat["lo"]).fillna(0)

    # clip the lower CI at zero (counts can't be negative)
    dat["errm"] = np.minimum(dat["errm"], dat["mean"])  # ensures mean - errm >= 0

    fig = px.bar(
        dat,
        x="weekday_name", y="mean", color="scenario",
        facet_row="group",             # hip / ambulatory rows
        barmode="group",
        error_y="errp",                # <-- use column names
        error_y_minus="errm",
        category_orders={"weekday_name": WEEKDAYS},
        title=f"{kpi}: mean ±95% CI by weekday",
    )

    # presentation tweaks
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None, range=[0, None])     # never below zero
    fig.update_traces(error_y=dict(symmetric=False, width=4, thickness=1))  # small caps

    fig.show()



# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np

# A) KPIs per (scenario, rep)
df_kpis = pd.json_normalize(kpi_rows)

# pick the event-based breach total from KPIs
kpi_breaches = (
    "breach_incidence_total"
    if "breach_incidence_total" in df_kpis.columns
    else "breached_total"
)

# B) Build patients_df (one row per patient)
def patients_to_df(patients_by_scen):
    frames=[]
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

patients_df = patients_to_df(patients_by_scen).drop_duplicates(
    subset=["scenario","rep","id"]
).copy()

# C) Ensure 'breach_time' exists where needed (belt-and-braces)
if "breach_time" not in patients_df.columns:
    patients_df["breach_time"] = np.nan

need_bt = patients_df["breach_time"].isna() & patients_df["breached"].astype("boolean", errors="ignore").fillna(False)
patients_df.loc[need_bt, "breach_time"] = (
    patients_df.loc[need_bt, "arrival"].astype("Int64") + patients_df.loc[need_bt, "deadline"].astype("Int64")
)

# D) Define analysis window + masks
warmup_cut_min = cfg.get_warmup_cut_min()
end_time_min   = cfg.get_horizon_end_min()

served_mask  = patients_df["service_start"].notna() & (patients_df["service_start"] >= warmup_cut_min)

# Breach incidence: within window AND still waiting at breach_time
bt_in_window = patients_df["breach_time"].notna() & (patients_df["breach_time"] >= warmup_cut_min) & (patients_df["breach_time"] < end_time_min)
still_waiting_at_bt = patients_df["service_start"].isna() | (patients_df["service_start"] > patients_df["breach_time"])
breach_mask = bt_in_window & still_waiting_at_bt

# E) Patient-derived totals per (scenario, rep)
pt_served = (patients_df.loc[served_mask]
             .groupby(["scenario","rep"], observed=False)
             .size().rename("served_total_pt").reset_index())

pt_breach = (patients_df.loc[breach_mask]
             .groupby(["scenario","rep"], observed=False)
             .size().rename("breach_total_pt").reset_index())

# F) Compare to KPIs
cmp = (df_kpis[["scenario","rep",kpi_breaches,"served_total"]]
       .merge(pt_breach, on=["scenario","rep"], how="left")
       .merge(pt_served, on=["scenario","rep"], how="left")
       .fillna({"breach_total_pt":0, "served_total_pt":0}))

cmp["delta_breach"] = cmp["breach_total_pt"] - cmp[kpi_breaches]
cmp["delta_served"] = cmp["served_total_pt"] - cmp["served_total"]

print(cmp.sort_values(["scenario","rep"]).head(12))
print("\nMax |delta| breach, served:",
      cmp["delta_breach"].abs().max(), cmp["delta_served"].abs().max())

# strict invariants
assert (cmp["delta_breach"].abs().max() == 0), "Incidence mismatch vs patients"
assert (cmp["delta_served"].abs().max() == 0), "Served mismatch vs patients"
print("KPI ↔ patients consistency checks passed")



# In[ ]:


# Breach incidence should be <= arrivals - served (post-warmup window-wise, roughly)
# and should drop in higher-capacity scenarios:
print(cmp.groupby("scenario")[["breach_incidence_total","served_total"]].mean())

# End-of-horizon backlog should be smallest in 'more_capacity'
df_kpis = pd.json_normalize(kpi_rows)
print(df_kpis.groupby("scenario")["waiting_count_final"].mean().sort_values())


# In[ ]:





# ## 2. Backlog: waiting_count_final vs breached_waiting_final  (end of horizon)

# In[ ]:


kpis = ["waiting_count_final","breached_waiting_final"]
long = melt_kpis(df_scen, kpis)
px.box(long, x="scenario", y="value", color="scenario",
       facet_col="kpi", facet_col_wrap=2,
       title="End-of-horizon backlog & breached backlog").show()


# # 3. Utilisation 

# In[ ]:


kpis = ["utilisation_rate_post_warmup"]
long = melt_kpis(df_scen, kpis)
px.box(long, x="scenario", y="value", color="scenario",
       title="Utilisation rate (post-warmup)").show()


# In[ ]:


# check splits are correct
WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
AMB_KINDS = ("shoulder","wrist","ankle")

# capacity per (scenario,rep) from KPIs
cap = (df_scen[["scenario","rep","util_minutes_capacity_total_post_warmup",
                "utilisation_rate_post_warmup"]]
       .rename(columns={"util_minutes_capacity_total_post_warmup":"capacity"}))

# patients table (one row per patient)
def patients_to_df(pby):
    frames=[]
    for scen, rep_map in pby.items():
        for rep, plist in rep_map.items():
            if not plist: continue
            d = pd.DataFrame(plist)
            if d.empty: continue
            d["scenario"]=scen; d["rep"]=int(rep)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

patients_df = patients_to_df(patients_by_scen).drop_duplicates(subset=["scenario","rep","id"]).copy()

warm = cfg.get_warmup_cut_min()

served = patients_df.query("service_start.notna() and service_start >= @warm")\
                    .assign(group=lambda d: np.where(d["kind"].eq("hip"), "hip", "ambulatory"))

used_by_group = (served.groupby(["scenario","rep","group"], observed=False)["duration"]
                 .sum().rename("minutes_used").reset_index())

# contributions = minutes_used_g / total capacity
util_split = used_by_group.merge(cap, on=["scenario","rep"], how="left")\
                          .assign(util_contribution=lambda d: d["minutes_used"]/d["capacity"])

# check they sum to overall utilisation
sum_by_rep = (util_split.pivot(index=["scenario","rep"], columns="group", values="util_contribution")
                         .fillna(0.0))
sum_by_rep["sum_groups"] = sum_by_rep.sum(axis=1)

recon = sum_by_rep.join(cap.set_index(["scenario","rep"])["utilisation_rate_post_warmup"])
max_gap = (recon["sum_groups"] - recon["utilisation_rate_post_warmup"]).abs().max()
print(f"Max |sum(groups) - overall util| = {max_gap:.6f}")


# In[ ]:


import numpy as np
import pandas as pd

# 0) Make sure you actually have per-rep patients (run_scenarios(..., attach_patients_last_only=False))
def patients_to_df(patients_by_scen):
    frames=[]
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

patients_df = patients_to_df(patients_by_scen)

# 1) KPI-side minutes used (post-warmup)
df_scen = pd.json_normalize(kpi_rows)
kpi_used = (df_scen[["scenario","rep","util_minutes_used_total_post_warmup"]]
            .rename(columns={"util_minutes_used_total_post_warmup":"kpi_used"}))

# 2) Patient-side minutes used = sum of served durations post-warmup
warmup_cut_min = cfg.get_warmup_cut_min()

served_rows = patients_df.loc[
    patients_df["service_start"].notna() & (patients_df["service_start"] >= warmup_cut_min)
].copy()

# ensure numeric duration (minutes)
served_rows["duration"] = served_rows["duration"].astype(float)

pt_used = (served_rows
           .groupby(["scenario","rep"], observed=False)["duration"]
           .sum()
           .rename("pt_used")
           .reset_index())

# 3) Merge and compare
chk = (kpi_used.merge(pt_used, on=["scenario","rep"], how="outer")  # outer to spot gaps
               .fillna(0))                                         # 0 minutes for missing reps

# Use a tolerance (floating-point arithmetic)
tol = 1e-6
diff = (chk["pt_used"] - chk["kpi_used"]).abs()
print("max abs diff (min):", diff.max())

# If you want a hard assertion:
assert np.allclose(chk["pt_used"], chk["kpi_used"], atol=tol), "Mismatch between patient-sum and KPI minutes used."

print("passed")


# # 5. Surgical types served 

# In[ ]:


served_cols = [c for c in df_scen.columns if c.startswith("served_by_kind.")]
breached_cols = [c for c in df_scen.columns if c.startswith("breached_by_kind.")]

long_srv = melt_kpis(df_scen, served_cols)
long_srv["kpi"] = long_srv["kpi"].str.replace("served_by_kind.","")

px.box(long_srv, x="scenario", y="value", color="scenario",
       facet_col="kpi", facet_col_wrap=4,
       title="Served by kind (per replication)").show()

long_br = melt_kpis(df_scen, breached_cols)
long_br["kpi"] = long_br["kpi"].str.replace("breached_by_kind.","")
px.box(long_br, x="scenario", y="value", color="scenario",
       facet_col="kpi", facet_col_wrap=4,
       title="Breached (served) by kind (per replication)").show()


# ## 5. Time series utilisation

# In[ ]:


# explode the per-rep list of dicts into rows
u = (df_scen[["scenario","rep","utilisation_by_day_post_warmup"]]
       .explode("utilisation_by_day_post_warmup")
       .dropna())
u = pd.concat([u.drop(columns=["utilisation_by_day_post_warmup"]),
               u["utilisation_by_day_post_warmup"].apply(pd.Series)], axis=1)

# mean & CI by day & scenario
g = (u.groupby(["scenario","day"], as_index=False)
       .agg(mean=("utilisation","mean"), n=("utilisation","size"), se=("utilisation","sem")))
g["lo"] = g["mean"] - 1.96*g["se"]
g["hi"] = g["mean"] + 1.96*g["se"]

fig = px.line(g, x="day", y="mean", color="scenario",
              title="Utilisation over time (mean ±95% CI)")
# add ribbons
for scen in g["scenario"].unique():
    sub = g[g["scenario"]==scen]
    fig.add_traces(px.scatter(sub, x="day", y="hi").update_traces(visible=False).data)
    fig.add_traces(px.scatter(sub, x="day", y="lo").update_traces(visible=False).data)
fig.show()


# ## 6. Sctterplot breach by service
# 
# 'More capacity' scenario serves more with fewer breaches. Overall runs with higher throughput have fewer breaches (near saturation behaviour so small slack collapses waits)

# In[ ]:


px.scatter(df_scen, x="served_total", y="breach_incidence_total",
           color="scenario", hover_data=["rep"],
           trendline="ols", trendline_scope="trace",
           title="Throughput vs Breached (each dot = replication)").show()


# ## Breaches by weekday (counts and per-day average)

# ## Utilislation by weekday

# In[ ]:


WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def plot_util_by_weekday(df_scen: pd.DataFrame, *, show_points=True, ci=0.95, pad=0.01):
    """
    Plot utilisation by weekday (post-warmup) with mean ±95% CI per scenario.

    df_scen: pd.DataFrame from pd.json_normalize(kpi_rows)
             must contain ['scenario','rep','utilisation_by_day_post_warmup'] where each entry
             is a list of dicts: {'day', 'minutes_used', 'minutes_capacity', 'utilisation'}.

    show_points: overlay per-rep daily points to see spread.
    ci: confidence level for error bars (use 0.95).
    pad: extra padding for y-axis (fraction of 1.0).
    """
    # ---- 1) Flatten per-day utilisation into a tidy frame
    util_rows = []
    for _, r in df_scen[["scenario","rep","utilisation_by_day_post_warmup"]].iterrows():
        daylist = r["utilisation_by_day_post_warmup"]
        if not isinstance(daylist, list):
            continue
        rep = int(r["rep"])
        scen = r["scenario"]
        for d in daylist:
            day_idx = int(d["day"])
            util_rows.append({
                "scenario": scen,
                "rep": rep,
                "day": day_idx,
                "weekday_idx": day_idx % 7,
                "weekday_name": WEEKDAYS[day_idx % 7],
                "utilisation": float(d["utilisation"]),
            })
    util_by_day = pd.DataFrame(util_rows)
    if util_by_day.empty:
        raise ValueError("No per-day utilisation data found in 'utilisation_by_day_post_warmup'.")

    # ---- 2) Aggregate mean ± 95% CI (normal approx; fine for n≈30)
    g = (util_by_day
         .groupby(["scenario","weekday_name"], observed=False)["utilisation"])
    agg = (g.agg(mean="mean", n="size", sd="std").reset_index())
    z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96  # keep simple; swap to t if you want small-n correction
    agg["se"]  = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["lo"]  = (agg["mean"] - z * agg["se"]).clip(lower=0)
    agg["hi"]  = (agg["mean"] + z * agg["se"]).clip(upper=1)
    agg["errp"] = (agg["hi"] - agg["mean"]).fillna(0)
    agg["errm"] = (agg["mean"] - agg["lo"]).fillna(0)

    # Stable weekday order
    agg["weekday_name"] = pd.Categorical(agg["weekday_name"], categories=WEEKDAYS, ordered=True)
    util_by_day["weekday_name"] = pd.Categorical(util_by_day["weekday_name"], categories=WEEKDAYS, ordered=True)

    # ---- 3) Auto-zoom Y so you can actually see the bars
    ymin = float(max(0.0, agg["lo"].min() - pad))
    ymax = float(min(1.0, agg["hi"].max() + pad))
    if ymax - ymin < 0.02:          # ensure at least a 2% window
        mid = (ymin + ymax) / 2
        ymin = max(0.0, mid - 0.01)
        ymax = min(1.0, mid + 0.01)

    # ---- 4) Plot: mean line with CI error bars; optional raw points
    fig = px.line(
        agg,
        x="weekday_name", y="mean", color="scenario", markers=True,
        error_y="errp", error_y_minus="errm",
        title="Utilisation by weekday (mean ±95% CI, post-warmup)",
        labels={"weekday_name":"Weekday","mean":"Utilisation"},
    )
    fig.update_yaxes(range=[ymin, ymax], tickformat=".0%")

    if show_points:
        pts = px.scatter(
            util_by_day, x="weekday_name", y="utilisation", color="scenario",
            opacity=0.4, hover_data=["rep"],
        )
        for tr in pts.data:
            tr.showlegend = False
            fig.add_trace(tr)

    fig.show()
    return agg, util_by_day

# --- call it
# df_scen = pd.json_normalize(kpi_rows)
agg_util, util_by_day = plot_util_by_weekday(df_scen, show_points=True, ci=0.95, pad=0.01)


# In[ ]:


px.box(util_by_day, x="weekday_name", y="utilisation", color="scenario",
      category_orders={"weekday_name": WEEKDAYS},
      title="Utilisation by weekday (per-rep distribution)",
      labels={"weekday_name":"Weekday","utilisation":"Utilisation"}) \
 .update_yaxes(tickformat=".0%").show()


# In[ ]:


# qucik check on idle capacity - seems counterintuitive

WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def build_util_by_day(df_scen: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten df_scen['utilisation_by_day_post_warmup'] into a tidy table:
    one row per (scenario, rep, day).
    """
    required = {"scenario","rep","utilisation_by_day_post_warmup"}
    missing = required - set(df_scen.columns)
    if missing:
        raise KeyError(f"df_scen is missing: {sorted(missing)}")

    rows = []
    for _, r in df_scen[["scenario","rep","utilisation_by_day_post_warmup"]].iterrows():
        lst = r["utilisation_by_day_post_warmup"]
        if not isinstance(lst, list):
            # tolerate None/NaN gracefully
            continue
        scen = r["scenario"]
        rep  = int(r["rep"])
        for d in lst:
            # tolerate missing keys with .get(..., 0.0)
            day_idx   = int(d.get("day", 0))
            used_min  = float(d.get("minutes_used", 0.0))
            cap_min   = float(d.get("minutes_capacity", 0.0))
            util      = float(d.get("utilisation", (used_min / cap_min) if cap_min > 0 else 0.0))
            rows.append({
                "scenario": scen,
                "rep": rep,
                "day": day_idx,
                "weekday_idx": day_idx % 7,
                "weekday_name": WEEKDAYS[day_idx % 7],
                "minutes_used": used_min,
                "minutes_capacity": cap_min,
                "utilisation": util,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No per-day utilisation data found. Did the KPIs include 'utilisation_by_day_post_warmup'?")
    # stable weekday order
    out["weekday_name"] = pd.Categorical(out["weekday_name"], categories=WEEKDAYS, ordered=True)
    return out

# --- 1) Build util_by_day from your KPI rows
# df_scen = pd.json_normalize(kpi_rows)  # you likely already have this
util_by_day = build_util_by_day(df_scen)

# --- 2) Quick checks (SAFE versions) -----------------------------

# 2a) Any days with capacity but zero work?
idle = util_by_day.query("minutes_capacity > 0 and minutes_used == 0")
idle_counts = (idle.groupby("scenario", observed=False)["weekday_name"]
                   .value_counts()
                   .rename("idle_days")
                   .reset_index())
print("Idle operating days (count) by scenario & weekday:\n", idle_counts.head(20), "\n")

# 2b) Share of idle operating days per scenario
share_idle = (util_by_day.assign(
                    idle=(util_by_day["minutes_capacity"]>0) & (util_by_day["minutes_used"]==0))
              .groupby("scenario", observed=False)["idle"]
              .mean()
              .rename("share_idle_days"))
print("Share of operating days that were idle (per scenario):\n", share_idle, "\n")

# 2c) Sanity: capacity==0 days (should usually be 0 with your DOW plan)
cap0 = (util_by_day["minutes_capacity"] == 0).sum()
print("Days with zero capacity recorded:", int(cap0), "\n")

# 2d) Show a few idle examples to eyeball
examples = idle.sort_values(["scenario","rep","day"]).head(10)
print("Example idle days:\n", examples[["scenario","rep","day","weekday_name","minutes_capacity","minutes_used","utilisation"]], "\n")

# --- 3) Optional: weekly roll-up (smoother picture) -------------
wb = (util_by_day.assign(week=lambda d: (d["day"] // 7).astype(int))
      .groupby(["scenario","rep","week"], observed=False)
      .agg(minutes_used=("minutes_used","sum"),
           minutes_capacity=("minutes_capacity","sum"))
      .reset_index())
wb["weekly_util"] = np.where(wb["minutes_capacity"]>0,
                             wb["minutes_used"]/wb["minutes_capacity"], 0.0)

# Per-scenario weekly mean ±95% CI
wk = (wb.groupby(["scenario","week"], observed=False)["weekly_util"]
        .agg(mean="mean", n="size", sd="std").reset_index())
wk["se"] = wk["sd"] / np.sqrt(wk["n"].clip(lower=1))
wk["lo"] = (wk["mean"] - 1.96*wk["se"]).clip(lower=0)
wk["hi"] = (wk["mean"] + 1.96*wk["se"]).clip(upper=1)

print("Weekly utilisation (first few rows):\n", wk.head())


# In[ ]:


# days with capacity but no work (per scenario, weekday)
idle = util_by_day.query("minutes_capacity > 0 and minutes_used == 0")
idle.groupby("scenario")["weekday_name"].value_counts().sort_index()


# In[ ]:


# share of idle operating days
(util_by_day.assign(idle=lambda d: (d.minutes_capacity>0) & (d.minutes_used==0))
 .groupby("scenario")["idle"].mean().sort_index())


# In[ ]:


def plot_breached_served_per_day(
    df_in: pd.DataFrame,
    *,
    warmup_days: int | None = 28,
    kind_to_cat: dict | None = None,   # e.g. {"hip":"hip","shoulder":"amb","wrist":"amb","ankle":"amb"}
    rolling: int | None = 7,           # set None to disable smoothing
    scenario_col: str | None = "scenario",
    facet: bool = True,
    title: str = "Breached cases operated per day"
):
    """
    Plots # of served-in-breach per day (optionally 7-day rolling), split hip vs amb,
    optionally faceted by scenario. Returns (dataframe_used, figure).
    """

    df = df_in.copy()

    # --- Served only & derive service_day ---
    if "service_start" not in df.columns:
        raise ValueError("df_in must contain 'service_start'.")
    df = df[df["service_start"].notna()].copy()
    df["service_day"] = (df["service_start"] // 1440).astype(int)

    # --- Warmup filter (post-warmup only) ---
    if warmup_days is not None:
        df = df[df["service_start"] >= int(warmup_days) * 1440]

    # --- Breached-at-service flag (>= to match your model) ---
    need = {"arrival","deadline"}
    if "breached" not in df.columns:
        if not need.issubset(df.columns):
            missing = sorted(need - set(df.columns))
            raise ValueError(f"Need columns {missing} to infer 'breached'.")
        df["breached"] = (df["service_start"] - df["arrival"]) >= df["deadline"]

    # --- Category column hip vs amb ---
    if "kind" not in df.columns:
        raise ValueError("df_in must contain 'kind' to split hip vs amb.")
    if kind_to_cat is not None:
        df["cat"] = df["kind"].map(kind_to_cat).fillna("other")
    else:
        df["cat"] = np.where(df["kind"].eq("hip"), "hip", "amb")

    # --- Grouping keys ---
    group_cols = ["service_day","cat"]
    has_scen = bool(scenario_col) and (scenario_col in df.columns)
    if has_scen:
        group_cols.append(scenario_col)

    # --- Daily counts of breached-served ---
    by_service = (
        df.loc[df["breached"]]
          .groupby(group_cols, as_index=False, observed=False)
          .size()
          .rename(columns={"size":"breaches"})
          .sort_values(group_cols)
    )

    # --- Optional rolling mean ---
    ycol = "breaches"
    if rolling and rolling > 1:
        gb_cols = [c for c in group_cols if c != "service_day"]
        by_service["breaches_rm"] = (
            by_service.groupby(gb_cols, group_keys=False)["breaches"]
                      .apply(lambda s: s.rolling(int(rolling), min_periods=1).mean())
        ).values
        ycol = "breaches_rm"

    # --- Plot ---
    ttl = f"{title}{f' (rolling={rolling})' if rolling and rolling>1 else ''}"
    if has_scen and facet:
        fig = px.line(by_service, x="service_day", y=ycol,
                      color="cat",
                      facet_col=scenario_col, facet_col_wrap=3,
                      markers=True,
                      title=ttl,
                      labels={"service_day":"Day", ycol:"# breached served", "cat":"Group", scenario_col:"Scenario"})
        # Clean facet labels
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    elif has_scen:
        fig = px.line(by_service, x="service_day", y=ycol,
                      color=scenario_col, line_dash="cat",
                      markers=True,
                      title=ttl,
                      labels={"service_day":"Day", ycol:"# breached served", "cat":"Group", scenario_col:"Scenario"})
    else:
        fig = px.line(by_service, x="service_day", y=ycol,
                      color="cat", markers=True,
                      title=ttl,
                      labels={"service_day":"Day", ycol:"# breached served", "cat":"Group"})

    fig.update_traces(mode="lines+markers")
    return by_service, fig


# In[ ]:


def patients_to_df(patients_by_scen: dict) -> pd.DataFrame:
    """Flatten {scenario: {rep: [patient_dict,...]}} -> DataFrame."""
    frames = []
    for scen, rep_map in (patients_by_scen or {}).items():
        for rep, plist in (rep_map or {}).items():
            if not plist: 
                continue
            d = pd.DataFrame(plist)
            if d.empty:
                continue
            d["scenario"] = scen
            d["rep"] = int(rep)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# Rebuild df_scenarios (requires that you've already run run_scenarios(...) earlier)
df_scenarios = patients_to_df(patients_by_scen)

# Quick sanity check
print("df_scenarios shape:", df_scenarios.shape)
print("columns:", sorted(df_scenarios.columns)[:15], "...")


# In[ ]:


kind_to_cat = {"hip":"hip","shoulder":"amb","wrist":"amb","ankle":"amb"}
_ = plot_breached_served_per_day(
    df_scenarios,
    warmup_days=cfg.warmup_days,
    kind_to_cat=kind_to_cat,
    rolling=7,
    scenario_col="scenario",
    facet=True  # False to overlay scenarios
)


# In[ ]:


# Served only
dfp = df_scenarios.loc[df_scenarios["service_start"].notna()].copy()

# Post-warmup only
warmup_cut_min = cfg.get_warmup_cut_min()
dfp = dfp.loc[dfp["service_start"] >= warmup_cut_min].copy()

# Derive fields
dfp["service_day"] = (dfp["service_start"] // 1440).astype(int)
dfp["cat"]         = np.where(dfp["kind"].eq("hip"), "hip", "amb")
dfp["wait_days"]   = dfp["wait"] / (60*24)

# OPTIONAL: clip display at 99th pct for readability (doesn't change stats)
q_hi = dfp["wait_days"].quantile(0.99)
dfp["_wait_days_plot"] = dfp["wait_days"].clip(upper=q_hi)

fig = px.violin(
    dfp,
    y="_wait_days_plot", x="cat", color="cat",
    facet_col="scenario", facet_col_wrap=2,
    box=True, points=False,
    hover_data={"wait_days":":.2f", "cat":True, "scenario":True},
    title=f"Wait time of served patients (post-warmup){' — y clipped at 99th pct' if q_hi < dfp['wait_days'].max() else ''}",
    labels={"_wait_days_plot":"Wait (days)", "cat":"Group"}
)

# CLEAR matching (don’t pass a boolean)
fig.for_each_yaxis(lambda ax: ax.update(matches=None))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # tidy facet labels
fig.show()


# In[ ]:


served = (dfp.assign(service_day=(dfp["service_start"]//1440).astype(int))
              .groupby(["scenario","service_day"], as_index=False)
              .size().rename(columns={"size":"served"}))
served["cum"] = served.groupby("scenario")["served"].cumsum()

px.line(served, x="service_day", y="cum", color="scenario",
        title="Cumulative served patients by day").show()


# In[ ]:


warmup_cut_min = cfg.get_warmup_cut_min()
end_time_min   = cfg.get_horizon_end_min()

dfb = df_scenarios.copy()

# backfill breach_time where breached==True
if "breach_time" not in dfb.columns:
    dfb["breach_time"] = pd.NA
need_bt = dfb["breach_time"].isna() & dfb["breached"].astype("boolean")
dfb.loc[need_bt, "breach_time"] = (
    dfb.loc[need_bt, "arrival"].astype("Int64") + dfb.loc[need_bt, "deadline"].astype("Int64")
)

bt = dfb["breach_time"]
ss = dfb["service_start"]

# event definition: first time crossing SLA while still waiting
mask_event = (
    bt.notna()
    & ((ss.isna()) | (ss > bt))                # still waiting at breach time
    & (bt >= warmup_cut_min) & (bt < end_time_min)  # post-warmup, inside horizon
)

events = dfb.loc[mask_event, ["scenario","rep","kind","breach_time"]].copy()
events["breach_day"] = (events["breach_time"] // 1440).astype(int)

daily = (events.groupby(["scenario","breach_day"], observed=False)
               .size().rename("breaches").reset_index()
               .sort_values(["scenario","breach_day"]))
daily["cum"] = daily.groupby("scenario")["breaches"].cumsum()

fig = px.line(
    daily, x="breach_day", y="cum", color="scenario",
    markers=True,
    title="Cumulative breach events by day (post-warmup)",
    labels={"breach_day":"Day", "cum":"Cumulative breaches"}
)
fig.show()


# In[ ]:


events2 = events.copy()
events2["group"] = np.where(events2["kind"].eq("hip"), "hip", "amb")

daily_g = (events2.groupby(["scenario","group","breach_day"], observed=False)
                 .size().rename("breaches").reset_index()
                 .sort_values(["scenario","group","breach_day"]))
daily_g["cum"] = daily_g.groupby(["scenario","group"])["breaches"].cumsum()

fig = px.line(
    daily_g, x="breach_day", y="cum",
    color="group", facet_col="scenario", facet_col_wrap=2,
    markers=True,
    title="Cumulative breach events by day — hip vs ambulatory",
    labels={"breach_day":"Day","cum":"Cumulative breaches","group":"Group"}
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # clean facet labels
fig.show()


# In[ ]:


# events per (scenario, rep, day) from df_scenarios
ev = df_scenarios.copy()
ev["breach_day"] = (ev["breach_time"] // 1440).astype("Int64")
ev = ev[ev["breach_day"].notna()].copy()

# still-waiting-at-breach filter
mask = ev["service_start"].isna() | (ev["service_start"] > ev["breach_time"])
ev = ev[mask]

# post-warmup, within horizon
warmup_cut = cfg.get_warmup_cut_min(); end_min = cfg.get_horizon_end_min()
ev = ev[(ev["breach_time"] >= warmup_cut) & (ev["breach_time"] < end_min)]

daily = (ev.groupby(["scenario","rep","breach_day"], observed=False)
           .size().rename("breaches").reset_index())

# Moving average per scenario (averaged across reps)
ma = (daily.assign(breaches_ma=lambda d: d.groupby(["scenario","rep"])["breaches"]
                                   .transform(lambda s: s.rolling(7, min_periods=1).mean()))
           .groupby(["scenario","breach_day"], as_index=False)["breaches_ma"].mean())

import plotly.express as px
px.line(ma, x="breach_day", y="breaches_ma", color="scenario",
        title="Daily breach incidence (7-day MA, mean across reps)",
        labels={"breach_day":"Day","breaches_ma":"breaches/day"}).show()


# In[ ]:


# mean of last vx first 60 days post warmup (averge breaches)
last = daily[daily["breach_day"] > daily["breach_day"].max()-60]
first = daily[daily["breach_day"] <= daily["breach_day"].min()+60]
print(last.groupby("scenario")["breaches"].mean().rename("last60/day"))
print(first.groupby("scenario")["breaches"].mean().rename("first60/day"))


# In[ ]:


# mean of last 60 days to quantify steady state
tail = ma[ma["breach_day"] >= (ma["breach_day"].max()-60)]
tail_mean = tail.groupby("scenario")["breaches_ma"].mean().round(2)
print(tail_mean)  # breaches/day in steady state


# In[ ]:




