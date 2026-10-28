"""Quality control of the plane-of-array (POA) irradiance sensor against a clear-sky model and, when present,
a satellite or reanalysis POA series (signal ``poa_ref``).

Checks, in the order they run (each later check uses the corrections of the earlier ones):

1. **Clock** - for every day, the lag that best aligns the measured curve with the clear-sky curve. The median
   over clear days is the labelling offset (start/end/centre of the interval); runs of clear days whose lag
   departs from it by more than ``qc.shift_min`` are clock shifts (a run matching the DST season points to a
   time-zone mix-up).
2. **Orientation and level** - tilt, azimuth and a scale factor fitted to the clear-sky intervals (sun above
   ``qc.min_elevation``); the scale at the nominal orientation is the level against the clear-sky model.
3. **Shading** - on clear days, the measured-to-clear-sky ratio by sun azimuth at low sun; azimuth bins where
   the median ratio stays below ``qc.shading_ratio`` form a horizon mask for the sensor.
4. **Soiling and drift** - the sensor's daily level against three references: the clear-sky model on clear
   days, ``poa_ref``, and the plant itself (modelled / measured power). Each is divided by the same weeks of
   the first (baseline) year, so seasonal effects cancel, and the median of the three is used: aerosols move
   only the first, satellite errors only the second, plant soiling and degradation only the third. A decline
   followed by an abrupt recovery is sensor soiling; a sustained departure on which at least two references
   agree is drift. Drift within the baseline year cannot be seen this way; the level against ``poa_ref`` per
   year is reported for that reason.
5. **Gaps** - missing daylight values are filled from ``poa_ref`` or the clear-sky curve, scaled by the same
   day's measured-to-reference ratio.

Corrections are switched on per check in the config (``qc.correct_*``). Level and orientation are only
reported by default: rescaling a sensor to a model is a decision for the analyst.

Usage:
    from pv_budget import qc
    res = qc.run(cfg, data)
    res.poa, res.findings, res.daily
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import minimize

from pv_budget import model
from pv_budget.config import Config
from pv_budget.io import PlantData

log = logging.getLogger(__name__)


@dataclass
class QCResult:
    poa: pd.Series                      # corrected POA, W/m2
    flags: pd.DataFrame                 # per interval: clear_day, shifted, shaded, filled
    daily: pd.DataFrame                 # per local day: lag, ratio, clear flag, level, soiling, drift
    findings: list = field(default_factory=list)
    orientation: dict = field(default_factory=dict)
    shading: list = field(default_factory=list)
    clearsky: pd.Series | None = None   # clear-sky POA at the nominal orientation
    shading_stats: pd.DataFrame | None = None   # median measured/clear-sky ratio by sun azimuth, low sun


def finding(check, status, message, **values):
    return {"check": check, "status": status, "message": message, **values}


# ---------------------------------------------------------------- daily alignment with the clear-sky curve
def daily_alignment(poa: pd.Series, cs: pd.Series, elev: pd.Series, dt_h: float, tz: str, cfg) -> pd.DataFrame:
    step_min = dt_h * 60
    K = max(1, int(round(cfg.max_lag_min / step_min)))
    local = poa.index.tz_convert(tz)
    rows = []
    x_all, c_all, e_all = poa.to_numpy(), cs.to_numpy(), elev.to_numpy()
    for day, pos in pd.Series(np.arange(len(poa)), index=local).groupby(local.date):
        pos = pos.to_numpy()
        c = c_all[pos]
        sun = np.flatnonzero(c > 1)
        if len(sun) < 4:
            continue
        lo, hi = max(0, sun[0] - K), min(len(pos), sun[-1] + K + 1)
        x, c, e = x_all[pos][lo:hi], c[lo:hi], e_all[pos][lo:hi]
        if np.isnan(x).mean() > 0.1:
            continue
        x = np.nan_to_num(x)
        n = len(c)
        C = np.zeros((2 * K + 1, n))
        for j, L in enumerate(range(-K, K + 1)):
            if L >= 0:
                C[j, L:] = c[:n - L]
            else:
                C[j, :L] = c[-L:]
        xm = x - x.mean()
        Cm = C - C.mean(axis=1, keepdims=True)
        den = np.linalg.norm(Cm, axis=1) * np.linalg.norm(xm)
        corr = np.where(den > 0, Cm @ xm / np.where(den > 0, den, 1), -1.0)
        j = int(np.argmax(corr))
        lag = float(j - K)
        if 0 < j < 2 * K:
            a, b, d = corr[j - 1], corr[j], corr[j + 1]
            den = a - 2 * b + d
            if den < 0:
                lag += 0.5 * (a - d) / den
        cl = C[j]
        # clearness is judged on the upper half of the sun's path, so that a sensor shaded at low sun
        # still has clear days (the shading check needs them)
        mid = (cl > 0.05 * cl.max()) & (np.roll(e, j - K) > 0.5 * e.max())
        if mid.sum() < 3:
            mid = cl > 0.05 * cl.max()
        k = float(np.dot(x[mid], cl[mid]) / np.dot(cl[mid], cl[mid]))
        nrmse = float(np.sqrt(np.mean((x[mid] - k * cl[mid]) ** 2)) / max(np.mean(k * cl[mid]), 1e-9))
        c_mid = float(np.corrcoef(x[mid], cl[mid])[0, 1]) if x[mid].std() > 0 and cl[mid].std() > 0 else 0.0
        hi_sun = e > cfg.min_elevation
        ratio = float(x[hi_sun].sum() / c[hi_sun].sum()) if c[hi_sun].sum() > 0 else np.nan
        rows.append({"date": pd.Timestamp(day), "lag_min": lag * step_min, "corr": float(corr[j]), "k": k,
                     "nrmse": nrmse, "corr_mid": c_mid, "ratio": ratio, "h_meas": x.sum() * dt_h / 1000})
    d = pd.DataFrame(rows).set_index("date")
    d["clear"] = (d["nrmse"] < cfg.clear_nrmse) & (d["corr_mid"] > cfg.clear_min_corr - 0.01) & (d["corr"] > 0.97)
    return d


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def clock_check(daily: pd.DataFrame, dt_h: float, tz: str, cfg) -> tuple[float, list, list]:
    """Baseline lag (minutes) and clock-shift periods [(first_day, last_day, shift_min)]."""
    clear = daily[daily["clear"]]
    findings, periods = [], []
    if len(clear) < 10:
        findings.append(finding("clock", "info", "fewer than 10 clear days: clock not checked"))
        return 0.0, periods, findings
    base = float(clear["lag_min"].median())
    dev = (clear["lag_min"] - base).rolling(5, center=True, min_periods=3).median()
    step_min = dt_h * 60
    for i0, i1 in _runs((dev.abs() >= cfg.shift_min).to_numpy()):
        if i1 - i0 + 1 < cfg.shift_min_days:
            continue
        shift = float(np.median(clear["lag_min"].iloc[i0:i1 + 1] - base))
        shift = round(shift / step_min) * step_min
        # extend to the clear days just before/after that are not shifted
        first = clear.index[i0 - 1] + pd.Timedelta(days=1) if i0 > 0 else daily.index[0]
        last = clear.index[i1 + 1] - pd.Timedelta(days=1) if i1 + 1 < len(clear) else daily.index[-1]
        first = min(first, clear.index[i0])
        last = max(last, clear.index[i1])
        periods.append((first, last, shift))
    if abs(base) > step_min / 3:
        findings.append(finding("clock", "warn", f"timestamps sit {base:+.0f} min from the interval start the "
                                f"config declares (data.label); check the logger's labelling", lag_min=base))
    else:
        findings.append(finding("clock", "ok", f"labelling offset {base:+.1f} min", lag_min=base))
    for first, last, shift in periods:
        dst = ""
        try:
            days = pd.date_range(first, last, freq="D").tz_localize(tz)
            in_dst = np.mean([bool(d.dst()) for d in days])
            if in_dst > 0.8 and abs(abs(shift) - 60) < step_min:
                dst = " - matches the summer-time season: local time with DST read as standard time?"
        except (TypeError, AttributeError):
            pass
        findings.append(finding("clock", "fail", f"clock shifted by {shift:+.0f} min from {first:%Y-%m-%d} to "
                                f"{last:%Y-%m-%d}{dst}", shift_min=shift, start=f"{first:%Y-%m-%d}",
                                end=f"{last:%Y-%m-%d}"))
    return base, periods, findings


def apply_shifts(s: pd.Series, periods, dt_h: float, tz: str) -> tuple[pd.Series, np.ndarray]:
    out = s.copy()
    flag = np.zeros(len(s), bool)
    local_date = s.index.tz_convert(tz).normalize().tz_localize(None)
    for first, last, shift in periods:
        steps = int(round(shift / (dt_h * 60)))
        m = (local_date >= first) & (local_date <= last)
        out[m] = s.shift(-steps)[m]
        flag |= np.asarray(m)
    return out, flag


# ---------------------------------------------------------------- orientation
def _iso_poa(tilt, az, zen, sazi, dni, ghi, dhi, albedo):
    t, a = np.radians(tilt), np.radians(az)
    z, s = np.radians(zen), np.radians(sazi)
    cos_aoi = np.cos(z) * np.cos(t) + np.sin(z) * np.sin(t) * np.cos(s - a)
    return dni * np.clip(cos_aoi, 0, None) + dhi * (1 + np.cos(t)) / 2 + ghi * albedo * (1 - np.cos(t)) / 2


def fit_orientation(x, day, zen, sazi, dni, ghi, dhi, albedo, nominal_az) -> dict:
    """Tilt and azimuth that best explain the shape of each clear day (one scale per day, so day-to-day
    turbidity does not pull the fit), then one overall scale at that orientation."""
    day = np.asarray(day)
    nd = int(day.max()) + 1

    def score(tilt, az):
        y = _iso_poa(tilt, az, zen, sazi, dni, ghi, dhi, albedo)
        k = np.bincount(day, x * y, nd) / np.maximum(np.bincount(day, y * y, nd), 1e-9)
        return float(np.sqrt(np.mean((x - k[day] * y) ** 2)))

    best = (np.inf, 0.0, 0.0)
    for tilt in np.arange(0, 61, 2.0):
        for az in np.arange(nominal_az - 60, nominal_az + 61, 4.0):
            r = score(tilt, az)
            if r < best[0]:
                best = (r, tilt, az)
    res = minimize(lambda v: score(v[0], v[1]), x0=[best[1], best[2]], method="Nelder-Mead",
                   options={"xatol": 0.1, "fatol": 1e-4})
    tilt, az = float(np.clip(res.x[0], 0, 90)), float(res.x[1])
    y = _iso_poa(tilt, az, zen, sazi, dni, ghi, dhi, albedo)
    return {"tilt": tilt, "azimuth": az, "scale": float(np.dot(x, y) / np.dot(y, y)), "rmse": score(tilt, az)}


# ---------------------------------------------------------------- level history (soiling, drift)
def _seasonal_level(series: pd.Series, first: pd.Timestamp, window: int = 20) -> pd.Series:
    """Series / its own same-season median in the baseline year (first 365 days)."""
    s = series.dropna()
    base = s[s.index < first + pd.Timedelta(days=365)]
    if len(base) < 30:
        return pd.Series(np.nan, index=series.index)
    bd, bv = base.index.dayofyear.to_numpy(), base.to_numpy()
    curve = np.full(366, np.nan)
    for d in range(1, 367):
        dist = np.minimum(np.abs(bd - d), 366 - np.abs(bd - d))
        sel = dist <= window
        if sel.sum() >= 3:
            curve[d - 1] = np.median(bv[sel])
    curve = pd.Series(curve).interpolate(limit_direction="both").to_numpy()
    return series / curve[series.index.dayofyear.to_numpy() - 1]


def level_history(daily: pd.DataFrame, q, findings: list) -> pd.DataFrame:
    """Combine the sensor's level against up to three references and look for soiling and drift.

    References (each divided by its own baseline-year season): clear-sky ratio on clear days, ratio to
    ``poa_ref`` (days with some sun), and the plant's modelled-to-measured power ratio. Each reference has its
    own confounders (aerosols, satellite errors, plant soiling and degradation), so a change counts only in the
    median of the references, and only when at least two of them move the same way.
    """
    cols = [c for c in ("ratio_cs", "ratio_ref", "ratio_plant") if c in daily and daily[c].notna().sum() > 60]
    if not cols:
        findings.append(finding("drift", "info", "no reference series long enough: drift not checked"))
        return daily
    first = min(daily[c].dropna().index[0] for c in cols)
    if (daily.index[-1] - first).days < 500:
        findings.append(finding("drift", "info", "less than ~1.4 years of data: drift not checked"))
        return daily
    lv = pd.DataFrame({c: _seasonal_level(daily[c], first) for c in cols})
    smooth = lv.apply(lambda s: s.dropna().rolling(15, center=True, min_periods=5).median().reindex(lv.index))
    smooth = smooth.interpolate(limit_area="inside")   # slow signals: bridge cloudy spells
    comb = smooth.median(axis=1, skipna=True)
    sign = np.sign(comb - 1)
    beyond = (smooth - 1).mul(sign, axis=0) > 0.75 * q.drift_flag
    agree = beyond.sum(axis=1)
    if "ratio_ref" in beyond:   # the satellite reference is the steadiest: where it exists, it must agree
        agree = agree.where(beyond["ratio_ref"] | smooth["ratio_ref"].isna(), 0)
    daily = daily.copy()
    for c in cols:
        daily[f"level_{c[6:]}"] = smooth[c]
    daily["level"] = comb
    # soiling: a decline then a sustained step up (cleaning or rain) in the combined level
    vals, days = comb.to_numpy(), comb.index
    soil = pd.Series(0.0, index=daily.index)
    episodes, n, k = [], 7, 7
    while k < len(vals) - n:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            before, after = np.nanmedian(vals[k - n:k]), np.nanmedian(vals[k:k + n])
            drop = np.nanmedian(vals[max(0, k - 4 * n):k - 2 * n]) - before if k >= 3 * n else 0.0
            # each reference on its own must see at least half the step (two of them at least)
            own = [np.nanmedian(smooth[c].to_numpy()[k:k + n]) / np.nanmedian(smooth[c].to_numpy()[k - n:k]) - 1
                   for c in cols]
        if np.isfinite(before) and np.isfinite(after) and after / before - 1 > q.soiling_step and \
                drop > 0.7 * q.soiling_step and sum(o > 0.5 * q.soiling_step for o in own) >= min(2, len(cols)):
            j = k - 1
            while j > 0 and np.isfinite(vals[j]) and vals[j] < after - 0.5 * (after - before) and \
                    (days[k] - days[j]).days < 150:
                j -= 1
            start, end = days[j], days[k - 1]
            seg = (daily.index >= start) & (daily.index <= end)
            span = max((end - start).days, 1)
            soil[seg] = np.maximum(soil[seg], (daily.index[seg] - start).days / span * (1 - before / after))
            episodes.append((start, days[k], 1 - before / after))
            k += 2 * n
        else:
            k += 1
    for s0, s1, mag in episodes:
        findings.append(finding("sensor_soiling", "warn", f"sensor soiling up to {mag:.1%} from {s0:%Y-%m-%d}, "
                                f"recovered {s1:%Y-%m-%d}", start=f"{s0:%Y-%m-%d}", end=f"{s1:%Y-%m-%d}",
                                loss=float(mag)))
    if not episodes:
        findings.append(finding("sensor_soiling", "ok", "no decline-and-recovery pattern beyond "
                                f"{q.soiling_step:.1%}"))
    corrected = comb / (1 - soil)
    after_base = corrected[corrected.index >= first + pd.Timedelta(days=365)]
    sustained = ((after_base - 1).abs() > q.drift_flag) & (agree.reindex(after_base.index) >= min(2, len(cols)))
    runs = [(a, b) for a, b in _runs(sustained.to_numpy())
            if (after_base.index[b] - after_base.index[a]).days >= 60]
    drift = pd.Series(1.0, index=daily.index)
    if runs:
        onset = after_base.index[runs[0][0]]
        pre = corrected[corrected.index <= onset]
        crossed = pre[(pre - 1).abs() <= q.drift_flag / 3]
        if len(crossed):
            onset = crossed.index[-1]
        full = corrected.interpolate(limit_direction="both")
        drift[daily.index >= onset] = full[daily.index >= onset]
        dev = after_base - 1
        peak = float(dev.loc[dev.abs().idxmax()])
        end_level = float(corrected.dropna().iloc[-1] - 1)
        findings.append(finding("drift", "fail", f"sensor level change from {onset:%Y-%m-%d}: {peak:+.1%} at most, "
                                f"{end_level:+.1%} at the end, against the baseline year ({len(cols)} references)",
                                start=f"{onset:%Y-%m-%d}", peak=peak, end_level=end_level))
    else:
        findings.append(finding("drift", "ok", f"no sustained level change beyond {q.drift_flag:.1%} against the "
                                f"baseline year ({len(cols)} references: {', '.join(c[6:] for c in cols)})"))
    daily["sensor_soiling"] = soil
    daily["drift"] = drift
    factor = pd.Series(1.0, index=daily.index)
    if q.correct_soiling:
        factor = factor * (1 - soil)
    if q.correct_drift:
        factor = factor * drift
    daily["factor"] = factor
    return daily


def plant_ratio(cfg: Config, data: PlantData, poa: pd.Series, tz: str) -> pd.Series:
    """Per local day: modelled (with the assumed degradation) / measured plant power on high-sun intervals with
    every inverter running. Plant soiling and a wrong degradation assumption leak into it, hence the consensus."""
    p = cfg.plant
    met = data.met
    t_amb = met["t_amb"].interpolate(limit=8).to_numpy() if "t_amb" in met else np.full(len(poa), 20.0)
    tc = model.cell_temperature(p.__class__(**{**p.__dict__, "temperature_source": "ambient"}), poa.to_numpy(),
                                t_amb, met["wind"].to_numpy() if "wind" in met else None)
    unc, _ = model.ac_power(p, poa.to_numpy(), tc, model.degradation(p, poa.index))
    pac0 = model.ac_limits(p)
    inv = [i.id for i in p.inverters]
    P = data.power.reindex(columns=inv).to_numpy()
    ok = (poa.to_numpy() > 400) & np.all(P > 0.05 * pac0[inv].to_numpy(), axis=1) & \
        np.all(unc[inv].to_numpy() < 0.95 * pac0[inv].to_numpy(), axis=1)
    if data.setpoint is not None:
        ok &= (data.setpoint.reindex(columns=inv).fillna(1.0).to_numpy() > 0.999).all(axis=1)
    day = poa.index.tz_convert(tz).normalize().tz_localize(None)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = pd.Series(np.where(ok, unc.sum(axis=1).to_numpy() / np.nansum(P, axis=1), np.nan), index=poa.index)
    g = r.groupby(day)
    return g.median().where(g.count() >= max(3, int(round(1.0 / data.dt_h))))


# ---------------------------------------------------------------- run
def run(cfg: Config, data: PlantData, sp: pd.DataFrame | None = None) -> QCResult:
    p, q = cfg.plant, cfg.qc
    tz = p.timezone
    idx = data.index
    sp = model.solar_position(idx, data.dt_h, p) if sp is None else sp
    loc = pvlib.location.Location(p.latitude, p.longitude, altitude=p.altitude)
    mid = model.midpoints(idx, data.dt_h)
    csc = loc.get_clearsky(mid, model="ineichen", solar_position=sp.set_index(mid))
    csc.index = idx
    cs = model.poa_from_components(p, sp, csc["ghi"], csc["dni"], csc["dhi"])
    elev = sp["apparent_elevation"]
    raw = data.met["poa"].astype(float).clip(lower=0) if "poa" in data.met else pd.Series(np.nan, index=idx)
    findings = []
    flags = pd.DataFrame(False, index=idx, columns=["clear_day", "shifted", "shaded", "filled"])
    if raw.notna().sum() < 96:
        findings.append(finding("data", "fail", "no POA sensor data: reference used throughout"))
        poa = _fill(raw, cs, data, tz, flags)
        return QCResult(poa, flags, pd.DataFrame(), findings, clearsky=cs)

    # 1. clock
    daily0 = daily_alignment(raw, cs, elev, data.dt_h, tz, q)
    base, periods, f = clock_check(daily0, data.dt_h, tz, q)
    findings += f
    poa = raw
    if periods and q.correct_shift:
        poa, flags["shifted"] = apply_shifts(raw, periods, data.dt_h, tz)
        daily = daily_alignment(poa, cs, elev, data.dt_h, tz, q)
    else:
        daily = daily0.copy()
    daily["lag_raw_min"] = daily0["lag_min"].reindex(daily.index)
    local_day = idx.tz_convert(tz).normalize().tz_localize(None)
    clear_days = pd.DatetimeIndex(sorted(daily.index[daily["clear"]]))
    flags["clear_day"] = local_day.isin(clear_days)
    findings.append(finding("clear_days", "info", f"{len(clear_days)} clear days of {len(daily)}",
                            n=len(clear_days)))

    # 2. orientation and level vs clear sky
    m = flags["clear_day"].to_numpy() & (elev.to_numpy() > q.min_elevation) & poa.notna().to_numpy()
    orient = {}
    if m.sum() > 200:
        sel = np.flatnonzero(m)
        sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 6000)).astype(int)]
        day_id = pd.factorize(local_day[sel])[0]
        args = (sp["apparent_zenith"].to_numpy()[sel], sp["azimuth"].to_numpy()[sel], csc["dni"].to_numpy()[sel],
                csc["ghi"].to_numpy()[sel], csc["dhi"].to_numpy()[sel], p.albedo)
        x = poa.to_numpy()[sel]
        orient = fit_orientation(x, day_id, *args, nominal_az=p.azimuth)
        y_nom = _iso_poa(p.tilt, p.azimuth, *args)
        orient["scale_nominal"] = float(np.dot(x, y_nom) / np.dot(y_nom, y_nom))
        orient.update(nominal_tilt=p.tilt, nominal_azimuth=p.azimuth)
        d_tilt = orient["tilt"] - p.tilt
        d_az = (orient["azimuth"] - p.azimuth + 180) % 360 - 180
        bad_az = abs(d_az) > q.orientation_azimuth_flag and max(orient["tilt"], p.tilt) > 8
        status = "warn" if abs(d_tilt) > q.orientation_tilt_flag or bad_az else "ok"
        findings.append(finding("orientation", status, f"fitted tilt {orient['tilt']:.1f} deg (nominal {p.tilt:g}), "
                                f"azimuth {orient['azimuth']:.1f} deg (nominal {p.azimuth:g})",
                                tilt=orient["tilt"], azimuth=orient["azimuth"]))
        lvl = orient["scale_nominal"] - 1
        findings.append(finding("level_clearsky", "warn" if abs(lvl) > q.level_flag else "ok",
                                f"level {lvl:+.1%} against the clear-sky model on clear days (model "
                                f"uncertainty a few %)", level=lvl))
        if status == "warn" and q.correct_orientation:
            # decompose at a physically plausible level (the clear-sky scale removed), then put it back
            k = orient["scale"]
            poa = _reorient(poa / k, sp, orient, p, data).clip(upper=1.3 * cs) * k
            findings.append(finding("orientation", "info", "POA transposed to the nominal plane (GTI-DIRINT)"))

    # 3. shading of the sensor at low sun
    shading = []
    shading_stats = pd.DataFrame(columns=["az", "rho", "days"])
    if orient and m.sum() > 200:
        reoriented = any(f["check"] == "orientation" and f["status"] == "info" for f in findings)
        tilt, azim = (p.tilt, p.azimuth) if reoriented else (orient["tilt"], orient["azimuth"])
        cs_fit = model.poa_from_components(p, sp, csc["ghi"], csc["dni"], csc["dhi"], tilt, azim).to_numpy()
        hi = flags["clear_day"].to_numpy() & (elev.to_numpy() > max(q.shading_max_elevation, 25))
        num = pd.Series(np.where(hi, poa.to_numpy(), 0.0), index=idx).groupby(local_day).sum()
        den = pd.Series(np.where(hi, cs_fit, 0.0), index=idx).groupby(local_day).sum()
        k_day = (num / den.where(den > 0)).reindex(local_day).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = poa.to_numpy() / (k_day * cs_fit)
        low = flags["clear_day"].to_numpy() & (elev.to_numpy() > 2) & \
            (elev.to_numpy() < q.shading_max_elevation) & np.isfinite(rho)
        frame = pd.DataFrame({"az": sp["azimuth"].to_numpy()[low] // 5 * 5, "el": elev.to_numpy()[low],
                              "rho": rho[low], "day": local_day[low]})
        stats = frame.groupby("az").agg(rho=("rho", "median"), days=("day", "nunique"))
        # a dip below the ratio threshold, or well below the neighbouring azimuths (the clear-sky model is
        # less accurate at low sun, so the neighbours set the local baseline)
        ok = stats[stats["days"] >= 10]
        base = pd.Series({a: ok["rho"][(np.abs(ok.index - a) <= 30) & (np.abs(ok.index - a) > 10)].median()
                          for a in stats.index})
        rel = stats["rho"] / base
        stats["baseline"] = base
        bad = stats[((stats["rho"] < q.shading_ratio) | (rel < q.shading_ratio + 0.03)) & (stats["days"] >= 10)]
        for i0, i1 in _runs(np.isin(stats.index.to_numpy(), bad.index.to_numpy())):
            if i1 == i0 and min(stats["rho"].iloc[i0], rel.iloc[i0]) > q.shading_ratio - 0.1:
                continue   # a single marginal 5-degree bin is noise, not an obstacle
            az0, az1 = float(stats.index[i0]), float(stats.index[i1]) + 5
            cut = min(q.shading_ratio, float(np.nanmax(rel.iloc[i0:i1 + 1] * 1.0)) * float(
                np.nanmedian(base.iloc[i0:i1 + 1])) + 0.05)
            sub = frame[(frame["az"] >= az0) & (frame["az"] < az1) & (frame["rho"] < cut)]
            el_max = float(sub["el"].quantile(0.9)) if len(sub) else 0.0
            shading.append({"az_from": az0, "az_to": az1, "max_elevation": el_max,
                            "median_ratio": float(stats["rho"].iloc[i0:i1 + 1].median())})
            side = "morning" if az1 <= 180 else "evening" if az0 >= 180 else "midday"
            findings.append(finding("shading", "warn", f"{side} shading of the sensor: sun azimuth {az0:.0f}-"
                                    f"{az1:.0f} deg below {el_max:.0f} deg elevation (median ratio "
                                    f"{shading[-1]['median_ratio']:.2f})", az_from=az0, az_to=az1,
                                    max_elevation=el_max))
        if not shading:
            findings.append(finding("shading", "ok", "no recurring low-sun shading of the sensor"))
        az, el = sp["azimuth"].to_numpy(), elev.to_numpy()
        for s in shading:
            flags.loc[(az >= s["az_from"]) & (az < s["az_to"]) & (el <= s["max_elevation"] + 1) & (el > -1),
                      "shaded"] = True
        shading_stats = stats.reset_index()

    # 4. soiling and drift: the sensor's level against up to three references
    daily["ratio_cs"] = daily["ratio"].where(daily["clear"])
    ref = data.met["poa_ref"] if "poa_ref" in data.met else None
    if ref is not None and ref.notna().sum() > 1000:
        # sunny days only: on cloudy days satellite and ground see different clouds
        both = poa.notna() & ref.notna() & (elev > q.min_elevation) & ~flags["shaded"]
        num = poa.where(both).groupby(local_day).sum()
        den = ref.where(both).groupby(local_day).sum()
        csd = cs.where(both).groupby(local_day).sum()
        sunny = (den / csd > 0.75).reindex(daily.index).fillna(False) & daily["clear"]
        daily["ratio_ref"] = (num / den).reindex(daily.index).where(sunny)
    daily["ratio_plant"] = plant_ratio(cfg, data, poa, tz).reindex(daily.index)
    daily = level_history(daily, q, findings)
    if "factor" in daily and (q.correct_drift or q.correct_soiling):
        fac = daily["factor"].reindex(local_day).to_numpy()
        fac = np.where(np.isfinite(fac) & (fac > 0), fac, 1.0)
        poa = poa / fac

    # level against the reference series, per year
    if ref is not None and ref.notna().sum() > 1000:
        both = poa.notna() & ref.notna() & (elev > q.min_elevation)
        yr = idx.tz_convert(tz).year
        per_year = (poa[both].groupby(yr[both]).sum() / ref[both].groupby(yr[both]).sum())
        for y, r in per_year.items():
            findings.append(finding("level_reference", "warn" if abs(r - 1) > q.level_flag else "ok",
                                    f"{y}: level {r - 1:+.1%} against poa_ref", year=int(y), level=float(r - 1)))
        if q.correct_level:
            poa = poa / float(per_year.median())
            findings.append(finding("level_reference", "info", f"POA rescaled by 1/{per_year.median():.3f}"))
    elif q.correct_level and orient:
        poa = poa / orient["scale_nominal"]
        findings.append(finding("level_clearsky", "info", f"POA rescaled by 1/{orient['scale_nominal']:.3f}"))

    # 5. shaded intervals and gaps -> reference
    if q.correct_shading and flags["shaded"].any():
        poa = poa.mask(flags["shaded"])
    poa = _fill(poa, cs, data, tz, flags)
    n_fill = int((flags["filled"] & (cs > 20)).sum())
    if n_fill:
        findings.append(finding("gaps", "info", f"{n_fill} daylight intervals filled from the reference",
                                n=n_fill))
    return QCResult(poa.clip(lower=0), flags, daily, findings, orient, shading, cs, shading_stats)


def _reorient(poa, sp, orient, p, data):
    aoi = pvlib.irradiance.aoi(orient["tilt"], orient["azimuth"], sp["apparent_zenith"], sp["azimuth"])
    times = model.midpoints(sp.index, data.dt_h)
    comp = pvlib.irradiance.gti_dirint(pd.Series(poa.fillna(0).to_numpy(), times), pd.Series(aoi.to_numpy(), times),
                                       pd.Series(sp["apparent_zenith"].to_numpy(), times),
                                       pd.Series(sp["azimuth"].to_numpy(), times), times,
                                       orient["tilt"], orient["azimuth"], albedo=p.albedo)
    comp.index = sp.index
    out = model.poa_from_components(p, sp, comp["ghi"].fillna(0), comp["dni"].fillna(0), comp["dhi"].fillna(0))
    return out.where(poa.notna())


def _fill(poa: pd.Series, cs: pd.Series, data: PlantData, tz: str, flags: pd.DataFrame) -> pd.Series:
    """Missing daylight values from poa_ref (or clear sky) times the day's measured/reference ratio."""
    ref = data.met["poa_ref"] if "poa_ref" in data.met else None
    use = ref.where(ref.notna(), cs) if ref is not None else cs
    miss = poa.isna() & (cs > 0)
    if not miss.any():
        return poa.fillna(0.0)
    day = poa.index.tz_convert(tz).normalize()
    both = poa.notna() & use.notna() & (cs > 20)
    num = poa.where(both).groupby(day).sum()
    den = use.where(both).groupby(day).sum()
    ratio = (num / den).where(den > 50)
    typical = float(np.nanmedian(ratio)) if ratio.notna().any() else 1.0
    if ref is None:  # clear sky: an unknown day is an average day of its month, not a clear one
        typical = typical * 0.75
    r = ratio.reindex(day).to_numpy()
    r = np.where(np.isfinite(r), r, typical)
    filled = poa.copy()
    filled[miss] = (use * r)[miss]
    flags.loc[miss, "filled"] = True
    return filled.fillna(0.0)
