"""Plant, data, budget and method settings from one YAML file.

Every section has defaults, unknown keys are an error (a typo in a threshold should not pass silently).
See configs/*.yaml for complete examples.

Usage:
    from pv_budget.config import load_config
    cfg = load_config("configs/pl_demo.yaml")
    cfg.plant.dc_kwp, cfg.losses.daylight_wm2
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Inverter:
    id: str
    dc_kwp: float
    ac_kw: float


@dataclass
class Plant:
    name: str
    latitude: float
    longitude: float
    inverters: list[Inverter]
    altitude: float = 0.0
    timezone: str = "UTC"            # calendar of the monthly report
    tilt: float = 25.0
    azimuth: float = 180.0           # degrees from north, 180 = south
    albedo: float = 0.2
    gamma_pdc: float = -0.0035       # power temperature coefficient, 1/K
    temperature_model: str = "pvsyst"
    u_c: float = 29.0                # PVsyst-type thermal loss factors, W/m2K and W/m2K/(m/s)
    u_v: float = 0.0
    temperature_source: str = "ambient"   # ambient (same model as the budget) or module (back-sheet sensor)
    module_delta_t: float = 3.0      # cell minus back-of-module temperature at 1000 W/m2, K
    dc_loss: float = 0.08            # lumped DC losses (wiring, mismatch, LID, optical), fraction
    ac_loss: float = 0.01            # AC losses after the inverter (cables, transformer), fraction
    inverter_eta: float = 0.97       # nominal inverter efficiency
    cod: str | None = None           # commercial operation date, start of degradation
    degradation_per_year: float = 0.005

    @property
    def dc_kwp(self) -> float:
        return float(sum(i.dc_kwp for i in self.inverters))

    @property
    def ac_kw(self) -> float:
        return float(sum(i.ac_kw for i in self.inverters))


@dataclass
class Budget:
    path: str | None = None          # budget file; None = computed from PVGIS TMY with the plant model
    format: str = "hourly"           # hourly (PVsyst-like export) or monthly table
    sep: str = ";"
    columns: dict = field(default_factory=lambda: {
        "date": "date", "energy": "E_Grid", "poa": "GlobInc", "t_amb": "T_Amb", "wind": "WindVel",
        "clipping": "IL_Pmax"})
    date_format: str = "%d/%m/%y %H:%M"
    label: str = "start"             # time label of the hourly budget rows
    energy_unit: str = "kWh"
    # losses the budget assumed (fractions); scalar or 12 monthly values
    soiling: float | list = 0.01
    snow: float | list = 0.0
    availability: float = 0.99
    curtailment: float | list = 0.0
    reference_date: str | None = None   # date whose degradation the budget includes; None = COD (year-1 budget)
    tmy_cache: str | None = None


@dataclass
class Data:
    format: str = "long"             # long (timestamp, device_id, signal, value) or wide
    timezone: str = "UTC"            # of naive timestamps in the file; ignored when they carry an offset
    label: str = "start"             # start, end or center of the averaging interval
    interval: str | None = None      # e.g. "15min"; None = coarsest interval found in the file
    timestamp_column: str = "timestamp"
    signals: dict = field(default_factory=dict)   # long: {file signal: {signal, unit}}
    devices: dict = field(default_factory=dict)   # long: {file device id: plant inverter id or PLANT}
    columns: dict = field(default_factory=dict)   # wide: {column: {device, signal, unit}}
    plant_device: str = "PLANT"      # device id used for plant-level signals


@dataclass
class QC:
    reference: str = "auto"          # auto (poa_ref if present, else clear sky), clearsky, poa_ref
    min_elevation: float = 15.0      # degrees, for orientation and level checks
    clear_nrmse: float = 0.06        # a day is clear when POA follows the clear-sky shape this closely
    clear_min_corr: float = 0.995
    max_lag_min: float = 180.0
    shift_min: float = 30.0          # flag clock offsets larger than this
    shift_min_days: int = 5          # consecutive clear days needed for a clock-shift period
    drift_flag: float = 0.025        # flag a level change sustained >= 60 days against the baseline year
    soiling_step: float = 0.025      # step-up that marks a sensor cleaning
    orientation_tilt_flag: float = 5.0
    orientation_azimuth_flag: float = 10.0
    level_flag: float = 0.07
    shading_ratio: float = 0.85
    shading_max_elevation: float = 30.0
    correct_shift: bool = True
    correct_drift: bool = True
    correct_soiling: bool = True
    correct_shading: bool = True
    correct_orientation: bool = False
    correct_level: bool = False      # rescale to the reference level; needs a human decision


@dataclass
class Losses:
    daylight_wm2: float = 50.0
    reference_wm2: float = 300.0     # intervals used for the performance index
    reference_stability: float = 0.05   # ... where POA changes less than this share to both neighbours
    down_fraction: float = 0.01      # inverter below this share of AC rating counts as down
    min_expected_fraction: float = 0.05
    clip_margin: float = 0.97
    cleaning_min_step: float = 0.01
    cleaning_window_days: int = 5
    soiling_max_days: int = 90        # longest build-up attributed to one cleaning
    soiling_min_rate: float = 0.0002   # per day; a smaller typical rate of dry spells is taken as zero
    rain_cleaning_mm: float = 5.0
    snow_max_temp: float = 3.0      # daylight mean ambient temperature, degC
    snow_max_pi: float = 0.5
    snow_lookback_days: int = 7
    detect_plateaus: bool = False
    plateau_intervals: int = 4
    plateau_tolerance: float = 0.005


@dataclass
class Bridge:
    method: str = "sequential"       # sequential (loss accounting) or shapley


@dataclass
class Config:
    plant: Plant
    budget: Budget = field(default_factory=Budget)
    data: Data = field(default_factory=Data)
    qc: QC = field(default_factory=QC)
    losses: Losses = field(default_factory=Losses)
    bridge: Bridge = field(default_factory=Bridge)


def _build(cls, d: dict | None):
    d = dict(d or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)


def from_dict(d: dict) -> Config:
    d = dict(d)
    unknown = set(d) - {"plant", "budget", "data", "qc", "losses", "bridge"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    p = dict(d["plant"])
    p["inverters"] = [_build(Inverter, {**i, "id": str(i["id"])}) for i in p["inverters"]]
    for k in ("cod",):
        if p.get(k) is not None:
            p[k] = str(p[k])
    b = dict(d.get("budget") or {})
    if b.get("reference_date") is not None:
        b["reference_date"] = str(b["reference_date"])
    if "columns" in b:
        b["columns"] = {**Budget().columns, **b["columns"]}
    cfg = Config(plant=_build(Plant, p), budget=_build(Budget, b), data=_build(Data, d.get("data")),
                 qc=_build(QC, d.get("qc")), losses=_build(Losses, d.get("losses")),
                 bridge=_build(Bridge, d.get("bridge")))
    if cfg.bridge.method not in ("sequential", "shapley"):
        raise ValueError("bridge.method must be sequential or shapley")
    if cfg.data.label not in ("start", "end", "center"):
        raise ValueError("data.label must be start, end or center")
    if cfg.plant.temperature_source not in ("ambient", "module"):
        raise ValueError("plant.temperature_source must be ambient or module")
    return cfg


def load_config(path) -> Config:
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def monthly(value, month: int) -> float:
    """A scalar or a list of 12 monthly values, for calendar month 1-12."""
    if isinstance(value, (list, tuple)):
        if len(value) != 12:
            raise ValueError("monthly values need 12 entries")
        return float(value[month - 1])
    return float(value)
