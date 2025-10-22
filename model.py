#!/usr/bin/env python
# coding: utf-8
# VERSION 1 MODEL - NO ROLLING ROTA, NO ELECTIVES, NO DYNAMIC PI

# # Scheduling model for hips and ambulatory surgery
from __future__ import annotations
from dataclasses import dataclass, field, replace, is_dataclass
from typing import Dict, List, Deque, Any, Tuple, Callable, Optional, Iterable
import math
import copy
from collections import deque
from pathlib import Path
from sim_tools.distributions import Exponential, Normal

import numpy as np
import pandas as pd
import plotly.express as px
import itertools
import model

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




# ## Daily planner
# 
# Two simple scheduling rules are used 

# In[5]:


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
                if wait_so_far > dline:          # strict '>' matches serve-time logic
                    in_breach_start[kind] += 1
                    tr("breach@start", kind=kind, pid=pid, wait_so_far=wait_so_far, deadline=dline)
                if prev_day_start_min is not None:
                    breach_time = arr + dline
                    if prev_day_start_min < breach_time <= day_start_min:
                        new_breaches_today[kind] += 1
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
                # --- FAST-FORWARD TO NEXT ARRIVAL (if any) ---
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

                tr("No candidates left, breaking day loop",
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
            if breach_excess > 0:
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
                "breached": breach_excess > 0,
                "excess": int(breach_excess),
                "policy": pol,
            })

        return (minutes_budget, served_today, breached_today,
                waits_today, excess_today, patient_records,
                in_breach_start, new_breaches_today)


# # Sampling arrivals and surgical duration
# 
# Using sim-tools

# In[6]:


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


# In[7]:


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

# In[8]:


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
                breached_now = excess > 0
                if breached_now:
                    breached_waiting_final += 1
                    breached_waiting_minutes_final += int(excess)
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
                    "in_warmup": arrival_min < warmup_cut_min,
                    "policy": policy,
                })
        tr("End-of-horizon unserved snapshot",
           waiting_count_final=int(waiting_count_final),
           waiting_minutes_final=int(waiting_minutes_final),
           breached_waiting_final=int(breached_waiting_final),
           breached_waiting_minutes_final=int(breached_waiting_minutes_final))

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
        kpis = {
            "policy": policy,  # Service policy used ("edd" earliest-due-date, or "pi" priority index), for record-keeping.

            # Served patients (post-warmup totals, by kind)
            "served_by_kind": served_counts,                   # Number of patients served by kind (post-warmup).
            "breached_by_kind": breached_counts,               # Of the served, how many were already past deadline at service start (post-warmup).

            # Served totals (post-warmup)
            "served_total": int(sum(served_counts.values())),  # Total patients served across all kinds (post-warmup).
            "breached_total": int(sum(breached_counts.values())),  # Total served-in-breach across all kinds (post-warmup).

            # End-of-horizon (final) waiting stock (served+unserved context)
            "waiting_count_final": int(waiting_count_final),   # How many patients still waiting at the end of the horizon (not served).
            "waiting_minutes_final": int(waiting_minutes_final),  # Sum of (end_time_min - arrival) across all still-waiting patients.

            # End-of-horizon breach among the unserved
            "breached_waiting_final": int(breached_waiting_final),               # Count of still-waiting patients who are in breach at horizon end.
            "breached_waiting_minutes_final": int(breached_waiting_minutes_final), # Sum of (wait - deadline) for those in breach at horizon end.

            # Daily breach snapshots (summed across post-warmup days, by kind)
            "in_breach_start_total_by_kind": in_breach_start_tot,  # Each day at start (08:00), count already-in-breach; summed over post-warmup days.
            "new_breaches_total_by_kind": new_breaches_tot,        # During the day, patients who newly cross the deadline; summed post-warmup.
            "in_breach_end_total_by_kind": in_breach_end_tot,      # At day end (24:00), still-waiting & in-breach; summed post-warmup.

            # Utilisation (post-warmup only)
            "util_minutes_used_total_post_warmup": util_minutes_used_total,        # Sum of minutes actually used (post-warmup days).
            "util_minutes_capacity_total_post_warmup": util_minutes_capacity_total,# Sum of available minutes (post-warmup days).
            "utilisation_rate_post_warmup": utilisation_rate_post_warmup,          # Used / Capacity over post-warmup days (0..1).
            "utilisation_by_day_post_warmup": utilisation_by_day_post_warmup,      # Per-day series for plotting/diagnostics.
        }

        return {"run_id": run_id, "seed_used": base_seed, "kpis": kpis, "patients": patient_records}


# # Get a clean copy of config ready for scenarios

# In[9]:


def clone_cfg(cfg):
    """safe copy for scenario overrides."""
    return replace(cfg) if is_dataclass(cfg) else copy.deepcopy(cfg)

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

# In[10]:


def run_single(cfg, run_id: int = 1) -> dict:
    sim = Simulation(cfg)
    return sim.single_run(run_id=run_id)


# # Mulitple reps of base config

# In[11]:


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


# In[12]:


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

# In[13]:


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


# In[14]:


def scenarios_to_df(kpi_rows):
    try:
        import pandas as pd
        return pd.json_normalize(kpi_rows)
    except Exception:
        return None





