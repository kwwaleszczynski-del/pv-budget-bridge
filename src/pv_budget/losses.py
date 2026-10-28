"""Loss accounting per interval and inverter: from the potential of a clean, available, unconstrained plant down
to the energy the model expects to have been measured.

The chain (each step works on what the previous one left):

    potential  x0  model AC power without inverter limit, at measured POA and cell temperature, degraded
    soiling        x0 - x1, x1 = same with irradiance x soiling ratio (daily, from the performance index)
    snow           x1 - x2, x2 = same with irradiance x (1 - snow cover); cold days with a collapsed index
    clipping       x2 - min(x2, AC limit)
    availability   inverter at ~zero output in daylight while the model expects output (not snow, not a
                   zero set-point): the whole clipped expectation is lost
    curtailment    set-point (signal or grid_command event) below the clipped expectation
    data gaps      power not recorded: the expectation after curtailment is unmeasured

Soiling ratio: the daily performance index (median of measured / model over steady-sky, unclipped,
unconstrained intervals with all inverters running) recovers abruptly when the panels are cleaned (logged
cleaning, rain, or a detected step). The step gives the loss just before the cleaning; the build-up is taken
as linear from the previous cleaning candidate. Where rain cleans often, build-ups are washed off before they
show as a step: then the typical soiling rate of the dry spells (when most of them lose performance) times the
days since the last rain is used. A build-up that has not yet ended in a cleaning, at a site without a
consistent rate, is not counted (it stays in the residual).

Usage:
    from pv_budget import losses
    res = losses.run(cfg, data, poa)
    res.frame[["soiling", "availability", ...]]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pv_budget import model
from pv_budget.config import Config
from pv_budget.io import PlantData

log = logging.getLogger(__name__)

COMPONENTS = ["soiling", "snow", "clipping", "availability", "curtailment", "data_gaps"]


@dataclass
class LossResult:
    frame: pd.DataFrame            # plant totals per interval, kW: potential, components, expected, actual
    state: pd.DataFrame            # per inverter: up / down / gap / night
    daily: pd.DataFrame            # performance index, soiling ratio, snow flags per local day
    cleanings: list = field(default_factory=list)
    downtime: pd.DataFrame | None = None   # detected downtime periods with cause from the event log
    t_cell: pd.Series | None = None
    weights: pd.Series | None = None


def event_mask(events: pd.DataFrame, idx: pd.DatetimeIndex, inverters: list[str], categories) -> pd.DataFrame:
    """Boolean frame (interval x inverter): covered by an event of the given categories."""
    out = pd.DataFrame(False, index=idx, columns=inverters)
    ev = events[events["category"].isin(categories)] if len(events) else events
    for r in ev.itertuples():
        m = (idx >= r.start) & (idx < r.end)
        cols = inverters if r.device in ("*", "PLANT", "nan") else [r.device] if r.device in inverters else []
        for c in cols:
            out.loc[m, c] = True
    return out


def caps(data: PlantData, inverters: list[str], cfg: Config, power: pd.DataFrame, x_lim: pd.DataFrame,
         pac0: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allowed share of each inverter's AC limit, and where it came from (signal / event / detected)."""
    idx = data.index
    cap = pd.DataFrame(1.0, index=idx, columns=inverters)
    src = pd.DataFrame("", index=idx, columns=inverters)
    if data.setpoint is not None:
        sp = data.setpoint.reindex(columns=inverters).fillna(1.0)
        cap = np.minimum(cap, sp)
        src = src.mask(sp < 0.999, "signal")
    ev = data.events[data.events["category"] == "grid_command"] if len(data.events) else data.events
    for r in ev.itertuples():
        m = (idx >= r.start) & (idx < r.end)
        lim = 0.0 if pd.isna(r.limit) else float(r.limit)
        cols = inverters if r.device in ("*", "PLANT", "nan") else [r.device] if r.device in inverters else []
        for c in cols:
            cap.loc[m, c] = np.minimum(cap.loc[m, c], lim)
            src.loc[m & (src[c] == "").to_numpy(), c] = "event"
    if cfg.losses.detect_plateaus:
        n, tol = cfg.losses.plateau_intervals, cfg.losses.plateau_tolerance
        for c in inverters:
            share = power[c] / pac0[c]
            flat = share.rolling(n, center=True).std() < tol
            below = (share < cfg.losses.clip_margin) & (share > 0.05) & (x_lim[c] > power[c] * 1.05)
            m = flat & below & (cap[c] > 0.999)
            cap.loc[m, c] = share[m]
            src.loc[m, c] = "detected"
    return cap.clip(0, 1), src


def _build_up_start(d: pd.Timestamp, candidates: pd.DatetimeIndex, prev: pd.Timestamp | None,
                    max_days: int) -> pd.Timestamp:
    """Start of the soiling build-up that ended with the cleaning seen on day d: the last cleaning candidate
    (rain, logged cleaning) before the cluster of candidates that produced this cleaning, or the previous
    confirmed cleaning, but not more than max_days back."""
    start = d - pd.Timedelta(days=max_days)
    if prev is not None:
        start = max(start, prev)
    if len(candidates):
        c = candidates[candidates <= d]
        k = len(c) - 1
        while k > 0 and (c[k] - c[k - 1]).days <= 2:     # walk back over a multi-day rain spell
            k -= 1
        if k > 0:
            start = max(start, c[k - 1] + pd.Timedelta(days=1))
    return start


def soiling_ratio(pi: pd.Series, cleaning_days: set, cfg: Config) -> tuple[pd.Series, list]:
    """Daily soiling ratio from the performance index; returns the ratio and the confirmed cleanings.

    A cleaning is confirmed where the index steps up by more than ``losses.cleaning_min_step`` (median of
    ``cleaning_window_days`` valid days after vs before) on a candidate day (rain, logged cleaning) or, without
    candidates, anywhere the step also stands out of the index's own noise.

    Two parts add up: (1) where most dry spells lose performance, a typical soiling rate times the days since
    the last rain or cleaning (small build-ups that rain washes off before they show as a step); (2) for each
    confirmed cleaning, whatever its step (1 - before/after) exceeds that, built up linearly from the
    previous candidate, the usual assumption of soiling-interval methods.
    """
    L = cfg.losses
    w = L.cleaning_window_days
    valid = pi.dropna()
    vals, days = valid.to_numpy(), valid.index
    cand = pd.DatetimeIndex(sorted(cleaning_days)) if cleaning_days else pd.DatetimeIndex([])
    rate = soiling_rate(valid, cand, L)
    # without known candidates, a step must also stand out of the day-to-day noise of the index
    # (robust sd of a w-day median, x3)
    min_step = L.cleaning_min_step
    if not cleaning_days and len(vals) > 10:
        noise = 1.4826 * np.median(np.abs(np.diff(vals))) / np.sqrt(2)
        min_step = max(min_step, 3 * 1.25 * noise / np.sqrt(w) / max(np.median(vals), 1e-9))
    # a candidate cleaning belongs to the first valid day on or after it
    owners = {}
    if len(cand):
        for c, i in zip(cand, days.searchsorted(cand)):
            if i < len(days):
                owners.setdefault(int(i), c)
    excess = pd.Series(0.0, index=pi.index)
    confirmed, prev = [], None
    for k in range(w, len(valid) - w + 1):
        d = days[k]
        if len(cand) and k not in owners:
            continue
        before, after = np.median(vals[k - w:k]), np.median(vals[k:k + w])
        if after / before - 1 < min_step:
            continue
        if confirmed and (d - confirmed[-1]["date"]).days < w:
            continue
        event = owners.get(k, d)
        start = _build_up_start(event, cand, prev, L.soiling_max_days)
        step_loss = 1 - before / after
        extra = max(0.0, step_loss - rate * min((event - start).days, L.soiling_max_days))
        span = pi.index[(pi.index >= start) & (pi.index < event)]
        if len(span) and extra > 0:
            frac = (span - start).days.to_numpy() / max((event - start).days, 1)
            excess.loc[span] = np.maximum(excess.loc[span].to_numpy(), frac * extra)
        confirmed.append({"date": event, "step": float(after / before - 1), "loss_before": float(step_loss),
                          "build_up_from": start})
        prev = event
    loss = excess
    if rate > 0:   # the typical build-up of dry spells, restarted by every rain or cleaning
        marks = pd.DatetimeIndex(sorted(set(cand) | {c["date"] for c in confirmed}))
        if len(marks):
            pos = marks.searchsorted(pi.index, side="right") - 1
            age = np.where(pos >= 0, (pi.index - marks[np.maximum(pos, 0)]).days, L.soiling_max_days)
        else:
            age = np.full(len(pi), L.soiling_max_days)
        loss = excess + rate * np.clip(age, 0, L.soiling_max_days)
    return 1.0 - loss.clip(0, 0.5), confirmed


def soiling_rate(valid: pd.Series, candidates: pd.DatetimeIndex, L) -> float:
    """Typical soiling rate (fraction per day): the median slope of the index over dry spells of a week or more
    (a seasonal drift of the index goes both ways across spells and drops out of the median); 0 when small."""
    if len(candidates) < 3 or len(valid) < 30:
        return 0.0
    rates = []
    edges = list(candidates) + [valid.index[-1] + pd.Timedelta(days=1)]
    for a, b in zip(edges[:-1], edges[1:]):
        seg = valid[(valid.index > a + pd.Timedelta(days=1)) & (valid.index < b)]
        if len(seg) >= 7 and (seg.index[-1] - seg.index[0]).days >= 7:
            x = (seg.index - seg.index[0]).days.to_numpy(float)
            slope = np.polyfit(x, seg.to_numpy() / np.median(seg.to_numpy()), 1)[0]
            rates.append(-slope)
    if len(rates) < 5:
        return 0.0
    med = float(np.median(rates))
    return med if med > L.soiling_min_rate else 0.0


def run(cfg: Config, data: PlantData, poa: pd.Series) -> LossResult:
    p, L = cfg.plant, cfg.losses
    idx = data.index
    inv = [i.id for i in p.inverters]
    pac0 = model.ac_limits(p)
    met = data.met
    t_amb = met["t_amb"].to_numpy() if "t_amb" in met else None
    t_mod = met["t_mod"].to_numpy() if "t_mod" in met else None
    if t_amb is None and t_mod is None:
        log.warning("no temperature signal: 20 degC ambient assumed")
        t_amb = np.full(len(idx), 20.0)
    if t_amb is not None:
        t_amb = pd.Series(t_amb, index=idx).interpolate(limit=8).fillna(pd.Series(t_amb).median()).to_numpy()
    if p.temperature_source == "module" and t_mod is not None:
        t_mod = pd.Series(t_mod, index=idx).interpolate(limit=8).to_numpy()
        tc = model.cell_temperature(p, poa.to_numpy(), t_mod=t_mod)
        if t_amb is not None:
            alt = model.cell_temperature(p.__class__(**{**p.__dict__, "temperature_source": "ambient"}),
                                         poa.to_numpy(), t_amb, met["wind"].to_numpy() if "wind" in met else None)
            tc = np.where(np.isfinite(tc), tc, alt)
    else:
        tc = model.cell_temperature(p, poa.to_numpy(), t_amb, met["wind"].to_numpy() if "wind" in met else None)
    deg = model.degradation(p, idx)
    x0, x0l = model.ac_power(p, poa.to_numpy(), tc, deg)
    x0.index = x0l.index = idx
    power = data.power.reindex(columns=inv).astype(float)
    gap = power.isna()
    P = power.fillna(0.0)
    cap, cap_src = caps(data, inv, cfg, P, x0l, pac0)
    daylight = (poa > L.daylight_wm2).to_numpy()
    tz = p.timezone
    local_day = idx.tz_convert(tz).normalize().tz_localize(None)
    logged = event_mask(data.events, idx, inv, ["outage", "planned_maintenance", "force_majeure", "grid_outage"])

    # performance index on reference intervals
    running = (P.to_numpy() > L.down_fraction * pac0[inv].to_numpy()) & ~gap.to_numpy()
    # stable sky only: on broken-cloud intervals sensor and array see different clouds
    g = poa.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        steady = (np.abs(np.diff(g, prepend=np.nan)) / g < L.reference_stability) & \
                 (np.abs(np.diff(g, append=np.nan)) / g < L.reference_stability)
    ref = (g > L.reference_wm2) & steady & running.all(axis=1) & (cap.to_numpy() > 0.999).all(axis=1) & \
          (x0.to_numpy() < L.clip_margin * pac0[inv].to_numpy()).all(axis=1) & ~logged.to_numpy().any(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = pd.Series(np.where(ref, P.sum(axis=1) / x0.sum(axis=1), np.nan), index=idx)
    grp = ratio.groupby(local_day)
    min_n = max(2, int(round(1.0 / data.dt_h)))
    pi = grp.median().where(grp.count() >= min_n)

    # snow: cold days with a collapsed plant ratio, after solid precipitation where precipitation is known;
    # a cover lasts while the days stay cold and the ratio low. Logged outages take precedence.
    tser = pd.Series(t_amb if t_amb is not None else tc, index=idx)
    day_t = tser.where(daylight).groupby(local_day).mean()
    cold = day_t < L.snow_max_temp
    recent = pd.Series(True, index=day_t.index)
    if "precip" in met:
        solid = pd.Series(met["precip"].fillna(0).to_numpy() * (tser.to_numpy() < 1.0), index=idx)
        recent = solid.groupby(local_day).sum().rolling(L.snow_lookback_days + 1, min_periods=1).sum() > 0.5
    all_up = running.all(axis=1)
    free = daylight & (cap.to_numpy() > 0.999).all(axis=1)
    day_ratio = (pd.Series(np.where(free, P.sum(axis=1), 0), index=idx).groupby(local_day).sum() /
                 pd.Series(np.where(free, x0l.sum(axis=1), 0), index=idx).groupby(local_day).sum())
    clean_level = pi.rolling(31, center=True, min_periods=5).median().reindex(day_ratio.index).ffill().bfill()
    plant_logged = pd.Series(np.where(daylight, logged.to_numpy().all(axis=1), False), index=idx)         .groupby(local_day).mean() > 0.5
    candidate = (cold & (day_ratio < 0.9 * clean_level) & ~plant_logged).reindex(day_ratio.index).fillna(False)
    start = candidate & recent.reindex(day_ratio.index).fillna(False)
    snow_day = start.copy()
    for k in range(1, len(snow_day)):
        if not snow_day.iloc[k] and snow_day.iloc[k - 1] and candidate.iloc[k]:
            snow_day.iloc[k] = True
    pi_ns = pi.where(~snow_day.reindex(pi.index).fillna(False))

    # soiling ratio
    clean_days = set()
    if len(data.events):
        for r in data.events[data.events["category"] == "cleaning"].itertuples():
            clean_days.add(pd.Timestamp(r.start).tz_convert(tz).normalize().tz_localize(None))
    if "precip" in met:
        rain = pd.Series(met["precip"].fillna(0).to_numpy(), index=idx).groupby(local_day).sum()
        clean_days |= set(rain.index[rain >= L.rain_cleaning_mm] + pd.Timedelta(days=1))
        clean_days |= set(rain.index[rain >= L.rain_cleaning_mm])
    sr_day, cleanings = soiling_ratio(pi_ns, clean_days if ("precip" in met or clean_days) else set(), cfg)
    sr = sr_day.reindex(local_day).fillna(1.0).to_numpy()

    x1, _ = model.ac_power(p, poa.to_numpy(), tc, deg * sr)
    x1.index = idx
    # snow cover per interval on snow days: what the measured ratio says is missing, plant-wide
    snow = np.zeros(len(idx))
    sd = snow_day.reindex(local_day).fillna(False).to_numpy()
    if sd.any():
        lvl = clean_level.reindex(local_day).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            r = P.sum(axis=1).to_numpy() / np.minimum(x1, pac0[inv].to_numpy()).sum(axis=1).to_numpy()
        snow = np.where(sd & daylight & ~logged.to_numpy().any(axis=1),
                        np.clip(1 - r / np.where(lvl > 0, lvl, 1.0), 0, 1), 0.0)
        snow = np.nan_to_num(snow)
    x2, _ = model.ac_power(p, poa.to_numpy(), tc, deg * sr * (1 - snow))
    x2.index = idx
    lim = x2.clip(upper=pac0[inv], axis=1)

    down = (P.to_numpy() < L.down_fraction * pac0[inv].to_numpy()) & \
           (x2.to_numpy() > L.min_expected_fraction * pac0[inv].to_numpy()) & daylight[:, None] & \
           ~gap.to_numpy() & (cap.to_numpy() > 0.01)
    capped = np.minimum(lim.to_numpy(), cap.to_numpy() * pac0[inv].to_numpy())
    up = ~down & ~gap.to_numpy()
    frame = pd.DataFrame({
        "potential": x0.sum(axis=1).to_numpy(),
        "soiling": (x0 - x1).sum(axis=1).to_numpy(),
        "snow": (x1 - x2).sum(axis=1).to_numpy(),
        "clipping": (x2 - lim).sum(axis=1).to_numpy(),
        "availability": np.where(down, lim.to_numpy(), 0.0).sum(axis=1),
        "curtailment": np.where(~down, lim.to_numpy() - capped, 0.0).sum(axis=1),
        "data_gaps": np.where(gap.to_numpy(), capped, 0.0).sum(axis=1),
        "expected": np.where(up, capped, 0.0).sum(axis=1),
        "actual": P.where(~gap, 0.0).sum(axis=1).to_numpy(),
        "poa": poa.to_numpy(), "t_cell": tc, "t_amb": t_amb if t_amb is not None else np.nan,
        "soiling_ratio": sr, "snow_cover": snow,
        "daylight": daylight}, index=idx)
    w = pd.Series({i.id: i.dc_kwp for i in p.inverters}) / p.dc_kwp
    frame["up_w"] = (up & daylight[:, None]).astype(float) @ w[inv].to_numpy()
    frame["down_w"] = down.astype(float) @ w[inv].to_numpy()
    frame["curtailed"] = (cap.to_numpy() < 0.999).any(axis=1)
    for cat in ["planned_maintenance", "force_majeure", "grid_outage", "outage"]:
        m = event_mask(data.events, idx, inv, [cat]).to_numpy()
        frame[f"down_{cat}"] = np.where(down & m, lim.to_numpy(), 0.0).sum(axis=1)
        frame[f"downw_{cat}"] = (down & m).astype(float) @ w[inv].to_numpy()
        frame[f"excl_{cat}"] = (m & daylight[:, None] & ~gap.to_numpy()).astype(float) @ w[inv].to_numpy()
        frame[f"exclE_{cat}"] = np.where(m & ~gap.to_numpy(), capped + np.where(down, lim.to_numpy() - capped, 0),
                                         0.0).sum(axis=1)
    frame["curtailment_event"] = np.where(~down & (cap_src.to_numpy() == "event"), lim.to_numpy() - capped,
                                          0.0).sum(axis=1)
    frame["curtailment_detected"] = np.where(~down & (cap_src.to_numpy() == "detected"),
                                             lim.to_numpy() - capped, 0.0).sum(axis=1)
    state = pd.DataFrame(np.where(gap.to_numpy(), "gap", np.where(down, "down", np.where(
        daylight[:, None], "up", "night"))), index=idx, columns=inv)
    daily = pd.DataFrame({"pi": pi, "pi_no_snow": pi_ns, "soiling_ratio": sr_day.reindex(pi.index),
                          "snow_day": snow_day.reindex(pi.index), "day_ratio": day_ratio.reindex(pi.index)})
    downtime = _downtime_periods(state, data.events, frame, lim, down, inv, data.dt_h)
    return LossResult(frame, state, daily, cleanings, downtime, pd.Series(tc, index=idx), w)


def _downtime_periods(state, events, frame, lim, down, inv, dt_h) -> pd.DataFrame:
    rows = []
    for n, i in enumerate(inv):
        d = down[:, n]
        if not d.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, d.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            t0, t1 = state.index[a], state.index[b - 1] + pd.Timedelta(hours=dt_h)
            cat = "unlogged"
            if len(events):
                ev = events[(events["start"] < t1) & (events["end"] > t0) &
                            (events["device"].isin([i, "*", "PLANT"]))]
                if len(ev):
                    cat = ev["category"].iloc[0]
            rows.append({"inverter": i, "start": t0, "end": t1, "hours": (b - a) * dt_h,
                         "energy_kwh": float(lim[i].to_numpy()[a:b].sum() * dt_h), "category": cat})
    return pd.DataFrame(rows, columns=["inverter", "start", "end", "hours", "energy_kwh", "category"])
