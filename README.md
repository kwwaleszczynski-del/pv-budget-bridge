# PV budget bridge

How far was a PV plant from its P50 budget last month, and why? This tool answers that question in the
format of a monthly asset-management report:

- a waterfall from budget to measured production: irradiance, temperature, degradation, soiling, snow,
  clipping, availability, curtailment, data gaps and an unexplained rest, adding up exactly every month;
- PR, temperature-corrected PR and weather-corrected PR;
- availability and PR guarantee tests with exclusions and damages, set up in YAML;
- quality control of the irradiance sensor that all of the above depends on.

Each part is checked against faults of known size injected into real (NREL PVDAQ) and synthetic data.

**Interactive page:** https://kwwaleszczynski-del.github.io/pv-budget-bridge/

The input is a generic long or wide table, with the time zone and DST handled explicitly. The output is an
Excel workbook and an HTML page per plant and month, so the same code runs locally on any SCADA export.

> Working material. The PR definitions follow public descriptions of IEC 61724-1 and of NREL's
> weather-corrected PR. This is not a certified implementation of the standard. The contract clauses in
> `configs/contract_example.yaml` are illustrative and must be checked against the actual agreement.

## Results in one table

Four plants, every month of their data, bars as % of the budget over the whole period
(`results/examples.md`):

| plant | months | budget MWh | actual MWh | irradiance | temperature | degradation | soiling | snow | clipping | availability | curtailment | data gaps | unexplained | PR | PR_wc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PL-DEMO (synthetic, 1.32 MWp) | 48 | 5,737.3 | 5,688.3 | +4.7 % | -0.3 % | -1.3 % | +0.8 % | -1.3 % | +0.3 % | +0.2 % | -1.4 % | -0.2 % | -2.3 % | 0.790 | 0.793 |
| PVDAQ-34 (146.6 kWp) | 36 | 779.0 | 664.4 | -2.7 % | -0.7 % | -3.3 % | +0.6 % | -0.0 % | -0.0 % | -0.4 % | +0.0 % | -2.3 % | -5.9 % | 0.663 | 0.667 |
| PVDAQ-1276 (68.5 kWp) | 36 | 348.0 | 282.1 | -2.2 % | -0.7 % | -2.8 % | +1.5 % | -0.0 % | -0.0 % | -1.0 % | +0.0 % | -3.3 % | -10.4 % | 0.629 | 0.632 |
| PVDAQ-1278 (171.4 kWp) | 36 | 873.3 | 755.1 | -1.8 % | -0.7 % | -2.8 % | +1.3 % | -0.1 % | -0.1 % | -0.2 % | +0.0 % | -1.4 % | -7.8 % | 0.669 | 0.673 |

A positive loss bar means the plant lost less than the budget assumed (e.g. the budgets assume 2 % soiling
in Las Vegas and 1.5 % in Poland).

## Verification

Every fault is injected on its own. The verification measures the change of the relevant bar between the
injected run and the base run, so each plant's own unknown losses cancel. That change is compared with what
was taken out. `python -m pv_budget verify` writes `results/verification.json`.

**Bridge, losses of known size** (error of the recovered kWh; "net" = soiling + clipping bars, because soiling
on a clipping inverter partly disappears into clipping):

| injected | PL-DEMO | PVDAQ-34 | PVDAQ-1276 | PVDAQ-1278 |
|---|---|---|---|---|
| 3-day outage of one inverter (logged) | +5.6 % | -0.0 % | +0.9 % | +0.6 % |
| limit to 90 % of the AC rating, 10 sunniest days of a month | +8.5 % (2,059 kWh) | truth 0 kWh, found 56 kWh | +76 % (91 kWh) | +3.3 % (114 kWh) |
| 2 % soiling (linear to 4 % over the driest 30 days, then a logged cleaning), net | -41 % | -107 % | -11 % | +27 % |

The outage hardly moves other bars: the unexplained bar moves by under 1 % of the injection. The 90 % limit
moves the unexplained bar by 7 % of the injection on PL-DEMO and by a few tens of kWh on the PVDAQ roofs.

Soiling is the weak bar:

- On PVDAQ-34 the step at the cleaning shows as 1.6 % instead of 4 %. On that roof the daily performance index
  scatters by 2-3 % and swallows a 2 % monthly loss.
- On PL-DEMO part of the injected build-up is taken as the typical rain-washed soiling rate rather than as
  the excess the cleaning revealed.

A soiling station or a cleaned reference string is the right tool when soiling matters contractually.

**Synthetic plant, every loss against its truth** (48 months, 15-minute data; the truth is computed in the same
order of losses as the estimate):

| loss | truth MWh | estimate MWh | total error | monthly MAE |
|---|---|---|---|---|
| availability | 46.7 | 46.5 | -0.5 % | 8 kWh |
| curtailment (set-points) | 78.3 | 77.9 | -0.5 % | 16 kWh |
| data gaps | 14.4 | 13.4 | -7.2 % | 61 kWh |
| soiling (rain-washed) | 52.0 | 47.9 | -7.9 % | 284 kWh |
| snow | 112.1 | 101.1 | -9.9 % | 230 kWh |
| clipping | 5.4 | 7.5 | +40 % | 45 kWh |

Monthly production is 20-190 MWh, so the mean monthly errors are 0.01 % (availability) to 0.24 %
(soiling) of the month's energy. The clipping error is the model's: the synthetic plant has incidence-angle losses that
the configured model does not.

**Sensor QC, faults of known size:**

| fault | result |
|---|---|
| 1-hour clock shift of the POA logger (3-7 months) | found 4 / 4, +60 min; start within 0-3 days, end within 0-10 days. Without QC the bars of those months move by 3.7-5.5 % of the budget in sum; with QC by 0-0.2 % |
| pyranometer drift falling linearly to -3 % over a year | flagged 4 / 4, but late (once beyond 2.5 %: 53-288 days after the start of the ramp). Level after the ramp estimated at -2.3 % (PL-DEMO) and -5.1 to -5.4 % (PVDAQ). Irradiance bar error without QC -2.3 to -2.4 %, with QC -0.8 to +0.5 % |
| sensor turned 20 deg west (synthetic) | fitted azimuth 198.7 deg, tilt 26.5 deg (true 200 / 25), flagged |
| obstacle shading the sensor, azimuth 100-160 deg below 20 deg (synthetic) | flagged at 100-105 deg below 19 deg; the rest of the sector sees low sun only on winter mornings, with fewer than 10 clear days |
| sensor soiling, three 40-day build-ups to 5 % (synthetic) | 1 of 3 found (3.5 %) |
| clean data, false alarms | 1 of 4 runs (PVDAQ-1278, a -4 % level change in winter 2017/18, see below) |

**Sequential vs Shapley:** the two allocations differ by 0.21-0.26 % of the monthly budget per bar on average.
In single months the difference reaches 7.5 % (PL-DEMO, a month with irradiance far above the budget) and 5-20 %
(PVDAQ, months with large data gaps). The interaction terms are large exactly when one factor is far from 1.

## What the real data showed

Three rooftop systems of one school campus in Las Vegas (NREL PVDAQ 34, 1276, 1278, 2016-2018), all logging the
same met-station POA sensor:

- **Timestamps** sit 14 minutes late when read as interval starts: PVDAQ stamps the end of each 15-minute
  average (`data.label: end`; the clock check then reads -1 min).
- **The sensor reads high.** It is 19-23 % above the clear-sky model on clear days and 9.5-16.4 % above ERA5
  (per year). Rescaled to ERA5 (`qc.correct_level: true`, a decision written in the config). Without that,
  the irradiance bar would read +10-16 % and the unexplained bar -15-25 %.
- **The sensor's plane** fits 5-7 deg tilt: roofs B and D (5 deg), not roof A (11.2 deg). For A the
  4.6 deg difference stays under the 5 deg flag.
- **The ambient sensor sticks:** about 31,500 quarter-hours repeat one value for 6 h or more (0 degF in winter).
  These are dropped and filled from ERA5 2 m temperature. The back-of-module sensor reads far below the cells
  on this site (found in the companion repository pv-thermal-check), so cell temperature comes from ambient.
- **Availability** (time-based, daylight, capacity-weighted):
  - 95.6 % for system 34 in 2016 (the June 2016 inverter failure);
  - 95.5 % for system 1278 in 2017;
  - 91.8 % for system 1276 in 2018 (zero output from November);
  - 99.5-100 % otherwise.

  None of these outages is logged; the tool lists them as "unlogged".
- **The unexplained bar is -6 to -10 % of the budget.** PR 0.63-0.67 against a design-type budget PR of 0.76:
  these 2010-2011 rooftops underperform a generic model (8 % DC losses, 0.5 %/year degradation). The
  year-on-year trend of the daily performance index is noisy: system 34 +0.15 %/yr (CI -0.24 to +0.68),
  1276 +0.72 %/yr (+0.31 to +0.98), 1278 -0.64 %/yr (-1.32 to -0.23).
- **The one QC false alarm** (system 1278, -4 % in winter 2017/18) does not appear for systems 34 and 1276,
  which use the same sensor. It comes from 1278's own plant-power reference: one of its inverters had a bad 2017.
  Three references reduce false alarms, they do not remove them.

The synthetic plant (PL-DEMO) is a 1.32 MWp / 1.0 MW ground-mounted plant at 51.9 N 19.5 E on ERA5 weather
2021-2024. It runs in Europe/Warsaw local time with DST, with snow, rain-washed soiling, inverter faults,
planned maintenance, grid outages, grid-operator set-points, one force-majeure event and communication gaps.
Its true plant differs from the configured model in five ways:

- temperature coefficient -0.38 instead of -0.35 %/K;
- thermal loss factors;
- DC losses;
- an incidence-angle loss;
- degradation 0.6 instead of 0.5 %/yr.

That gives an unexplained bar of -2.3 %.

## Method

The PR family, the availability and PR guarantee tests (step-down, tolerance, damages, bonus) and the
Shapley allocation are computed by [res-asset-kpi](https://github.com/kwwaleszczynski-del/res-asset-kpi)
(`res_asset_kpi.pv`, `res_asset_kpi.contracts`), the KPI library extracted from this and five related projects.

### Budget (P50)

The repository computes the budget with pvlib from the PVGIS typical meteorological year (PVGIS-SARAH3 for
Poland, PVGIS-ERA5 for Las Vegas). The model chain is:

- Perez transposition;
- PVsyst-type cell temperature, $`T_c = T_a + \alpha G (1-\eta)/(U_c + U_v v)`$;
- PVWatts DC with a linear temperature coefficient and lumped DC losses;
- the PVWatts inverter curve per inverter, with its AC limit;
- AC losses.

The losses the budget assumes are configured, and the bridge measures against them:

- soiling, scalar or monthly;
- snow, monthly;
- availability;
- curtailment;
- degradation at a reference date.

`python -m pv_budget budget` writes the hourly result in a layout close to a PVsyst hourly CSV export:

- comment lines;
- a row of variable names (`GlobInc`, `T_Amb`, `IL_Pmax`, `E_Grid`, ...);
- a units row;
- `dd/mm/yy HH:mm` dates in standard time.

The same reader takes a real export (column names remapped in `budget.columns`) or a monthly P50 table
(`budget.format: monthly`). The reader was tested on files written by this tool only, so check it against your
own export. PVsyst itself is not used.

### Bridge (waterfall)

The default method, **sequential loss accounting**, runs each month in a fixed order.

**1-3. Weather and age**, multiplicative on the budget $`E_b`$. $`M_{25}`$ and $`M`$ are the model's AC energy
without the inverter limit and without any assumed loss, at 25 degC and at the actual cell temperature:

```math
r_{irr} = \frac{M_{25}(G_{act})}{M_{25}(G_{bud})}, \qquad
r_T = \frac{M(G_{act},T_{act})/M_{25}(G_{act})}{M(G_{bud},T_{bud})/M_{25}(G_{bud})}, \qquad
r_D = \frac{\bar D_{act}}{D_{bud}}
```

```math
\Delta_{irr} = E_b (r_{irr}-1), \quad \Delta_T = E_b r_{irr} (r_T-1), \quad \Delta_D = E_b r_{irr} r_T (r_D-1),
\quad E_w = E_b r_{irr} r_T r_D
```

**4. Losses.** Per interval and inverter, a chain from the potential $`x_0`$ (model AC power without limit, at
measured POA and cell temperature, degraded) gives sequential loss fractions $`\ell_k`$ in this order:

1. soiling;
2. snow;
3. clipping;
4. availability;
5. curtailment;
6. data gaps.

They are turned into kWh on the actual plant's potential, so a bar is energy really lost, not a model's
estimate of it:

```math
P_{act} = \frac{E_{act}}{\prod_k (1-\ell_k)}, \qquad
L_k = P_{act} \prod_{j<k}(1-\ell_j)\,\ell_k
```

The budget-assumed losses $`b_k`$ are spread over the weather-adjusted budget in the same order:

```math
X_b = \frac{E_w}{\prod_k (1-b_k)}, \qquad B_k = X_b \prod_{j<k}(1-b_j)\, b_k, \qquad \Delta_k = B_k - L_k
```

**5. Unexplained.** The remainder is the gap between the actual plant's potential and the potential the
budget implies for this weather:

```math
\Delta_{res} = E_{act} - E_b - \sum \Delta = P_{act} - X_b
```

**Why sequential and not Shapley.** Both are computed (`bridge.method` picks the headline; the page has a
switch). The sequential order puts the exogenous factors first: weather and age. The operator-related losses
are then measured on the weather that actually occurred. That makes each loss bar the measured kWh lost, net of
what the budget assumed, which is the quantity an availability or performance clause pays for, and each
bar can be recomputed by hand.

Shapley writes the month as a product $`E_{act} = E_b \prod_k f_k`$, where the factors are:

- the three weather ratios;
- $`(1-\ell_k)/(1-b_k)`$ for each loss;
- the residual.

Each bar is then the factor's Shapley value. This is order-free, but no bar equals a measured quantity. On these
data the two differ by 0.2-0.3 % of the monthly budget on average, so the choice matters only in extreme months.

**Loss detection** (`src/pv_budget/losses.py`):

- **Availability.** An inverter counts as down when it is below 1 % of its rating in daylight while the model
  expects at least 5 %, unless the set-point is zero or snow covers the plant. The lost energy is the clipped
  expectation. Logged events give the cause.
- **Curtailment.** The set-point signal, or a `grid_command` event with a limit, caps the clipped expectation.
  Flat plateaus without a command can optionally be detected.
- **Snow.**
  - A snow day is a cold day (daylight mean below 3 degC) with the plant ratio below 90 % of its clean level,
    after solid precipitation where precipitation is known.
  - A snow period lasts while the days stay cold and the ratio stays low.
  - On those days the missing share of the ratio is snow cover.
- **Soiling.** The daily performance index is the median of measured / model over steady-sky, unclipped,
  unconstrained intervals. Soiling has two parts:
  1. the typical soiling rate of dry spells times the days since the last rain or cleaning;
  2. for each confirmed cleaning (an index step on a rain day or a logged cleaning), whatever the step
     exceeds that, built up linearly from the previous candidate.

  A build-up that has not yet ended in a cleaning stays in the unexplained bar.

### PR

```math
PR = \frac{E_{AC}/P_0}{H_{POA}/G_{STC}}, \qquad
PR_{25} = \frac{E_{AC}}{\sum_t P_0 \frac{G_t}{G_{STC}} \left(1+\gamma (T_{c,t}-25)\right)\Delta t}, \qquad
PR_{wc} = \frac{E_{AC}}{\sum_t P_0 \frac{G_t}{G_{STC}} \left(1+\gamma (T_{c,t}-T_{typ})\right)\Delta t}
```

$`T_{typ}`$ is the irradiance-weighted cell temperature of the budget's typical year (Dierauf et al. 2013),
so the annual $`PR_{wc}`$ stays close to the plain PR while its monthly values stop swinging with the seasons.
Cell temperature comes from ambient (the same model as the budget) or from a back-of-module sensor plus a
conduction offset (`plant.temperature_source`).

### Guarantees (example clauses)

- **Availability.** It is the capacity-weighted share of daylight intervals with the inverter running.
  - Intervals without data are left out.
  - So are excluded event categories (force majeure, grid-operator instruction, grid outage).
  - Planned maintenance is left out up to an allowance of hours per period; beyond the allowance it counts
    as downtime.
  - An energy basis is available.
- **PR.** The guarantee uses the chosen PR variant over intervals above an irradiance threshold, with complete
  data and no excluded event.
  - The guaranteed value falls each contract year.
  - A measurement tolerance is deducted.
  - Below a minimum data share the test is inconclusive.

Shortfall energy and damages:

```math
E_{short} = (A_g - A)\, E_{exp}, \qquad E_{short} = (PR_{g,eff} - PR)\, Y_{ref} P_0, \qquad
\mathrm{LD} = \min(E_{short}\, c, \mathrm{cap})
```

The tests are checked against examples worked out by hand (`tests/test_kpi_contract.py`).

### Irradiance sensor QC

1. **Clock.** For each day the lag that best aligns the measured curve with the clear-sky curve is found.
   - The median over clear days is the labelling offset.
   - A run of at least 5 clear days more than 30 min away from it is a clock shift.
   - A run matching the DST season is reported as a time-zone mix-up.
   - The shift is corrected before anything else.
2. **Orientation and level.** Tilt and azimuth are fitted to clear days, with one scale per day so that
   day-to-day turbidity does not pull the fit. The level is the scale at the nominal plane against the clear-sky
   model, and the level per year against `poa_ref`. A transposition to the nominal plane (GTI-DIRINT) and a
   rescale are available, but they are off unless switched on in the config.
3. **Shading.** On clear days the code takes the measured / clear-sky ratio at low sun by 5 deg azimuth bins.
   A bin is flagged when the ratio falls below 0.85, or when it falls clearly below the neighbouring azimuths
   (the clear-sky model is less accurate at low sun).
4. **Soiling and drift.** The sensor's daily level is compared with three references, each divided by the
   same weeks of the first year:
   - the clear-sky model on clear days;
   - `poa_ref` on clear days;
   - the plant's modelled / measured power.

   Their median is used. Aerosols move only the first reference, satellite errors the second, and plant
   soiling or a wrong degradation assumption the third.
   - A decline followed by a step up is sensor soiling.
   - A departure beyond 2.5 % sustained for 60 days is drift, but only when two references agree and the
     satellite reference, where it exists, is one of them.
5. **Gaps.** Missing daylight values are filled from `poa_ref` (or clear sky) times the same day's ratio.

## Using it on your own data

```bash
python -m pv_budget run --config my_plant.yaml --input export.csv --events events.csv --contract contract.yaml --month 2024-06 --out out/my_plant
```

- **Long format:** `timestamp, device_id, signal, value`
  (see `examples/long_format_sample.csv`, two days across the autumn DST change).
  - Standard signals: `poa`, `poa_ref`, `t_amb`, `t_amb_ref`, `t_mod`, `wind`, `precip`, `setpoint`,
    `e_export` at plant level, and `p_ac` per inverter.
  - Other names and units are mapped in `data.signals` (see `configs/pvdaq_*.yaml`).
- **Wide format:** one column per signal, mapped in `data.columns` (template: `configs/wide_mapping_example.yaml`).
- **Intervals:** 5, 10 or 15 minutes, or 1 hour. Mixed intervals are averaged up or spread down to
  `data.interval`.
- **Time:**
  - Strings with a UTC offset are exact.
  - Naive local times are read in `data.timezone`.
  - The repeated autumn hour is resolved by file order within each device and signal.
  - The skipped spring hour and lonely repeats are dropped with a warning (all at night).
  - `data.label` says whether a stamp is the start, end or centre of its interval.
  - Monthly reports use the plant's calendar (`plant.timezone`).
- **Event log:** CSV with `start, end, device (inverter id or *), category, limit`
  (see `examples/events_sample.csv`). Categories:
  - `outage`;
  - `planned_maintenance`;
  - `force_majeure`;
  - `grid_command` (with `limit` as a share of the AC rating);
  - `grid_outage`;
  - `cleaning`.
- **Outputs:**
  - `report.xlsx`: Summary, Bridge with conditional formatting, a waterfall chart per reported month,
    Losses, Contract, QC, Downtime;
  - `report.html` with an interactive waterfall;
  - `monthly.csv`.
- Paths come only from arguments or `PV_BUDGET_DATA` / `PV_BUDGET_RESULTS`.

## Limitations

- **The model is PVWatts-level.**
  - It has no beam/diffuse split for measured POA, so incidence-angle and spectral effects sit in the DC loss
    and in the unexplained bar.
  - It has no string or MPPT level and no partial inverter availability (states are up, down or no data).
- **Soiling estimates need cleanings or rain to anchor them.** A build-up that has not yet ended in one is
  invisible, and on noisy plants a 2 % monthly loss is within the noise (see the verification).
- **Snow detection needs temperature and, ideally, precipitation.** Without precipitation, a cold day with an
  unlogged outage can be taken for snow.
- **Drift detection is limited:**
  - it cannot see a drift inside the first (baseline) year;
  - it needs about 1.4 years of data;
  - it flags late, once the drift is beyond 2.5 %;
  - natural year-to-year differences between the references are 1-3 %, the size of a -3 % drift.
- **Shading detection needs clear days at low sun.** Winter-only obstacles in a cloudy climate are found only
  in part.
- **Satellite reference data:** PVGIS hourly POA is ERA5 reanalysis for the Americas and ends in 2023 (SARAH3).
- **The examples are limited:** the PVDAQ systems are small rooftops on one campus sharing one sensor, and the
  Polish plant is synthetic.
- **Budget file format:** the PVsyst-like reader and writer follow the variable names of PVsyst exports. They
  are not tested against a file produced by PVsyst.
- **No certification:** the IEC 61724-1 definitions are implemented from public descriptions and are not
  certified. The guarantee clauses are examples. The tool gives working numbers, not legal or contractual
  conclusions.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pv_budget all --jobs 4
```

`prepare` downloads the following (about 70 MB):

- the PVDAQ daily files (or copies them from `--pvdaq-mirror`);
- PVGIS TMY and hourly POA;
- one Open-Meteo ERA5 request for Las Vegas (precipitation, 2 m temperature).

It expects the ERA5 file for the Polish point (Open-Meteo archive, 51.9 N 19.5 E, hourly 2021-2024: shortwave,
direct normal and diffuse radiation, 2 m temperature, 10 m wind, precipitation, snowfall) at
`data/era5/era5_51.90_19.50_2021_2024.parquet` or `--era5`. The full run takes about 12 minutes on 4 processes.
The 30 tests run offline in about 10 s.

## Data and licences

- **NREL PVDAQ:** Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., Jordan, D. (2021).
  Photovoltaic Data Acquisition (PVDAQ) Public Datasets. OEDI. https://doi.org/10.25984/1846021 (CC BY 4.0).
- **PVGIS** (TMY, hourly POA): European Commission, Joint Research Centre.
- **ERA5:** Hersbach et al. (2020), Copernicus Climate Change Service, via the Open-Meteo archive API.

## References

- Dierauf, T., Growitz, A., Kurtz, S., Becerra Cruz, J. L., Riley, E., Hansen, C. (2013). Weather-Corrected
  Performance Ratio. NREL/TP-5200-57991.
- IEC 61724-1, Photovoltaic system performance - Part 1: Monitoring (definitions summarised, not quoted).
- Dobos, A. P. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641.
- Marion, B. (2015). A model for deriving the direct normal and diffuse horizontal irradiance from the global
  tilted irradiance. Solar Energy 122.
- Deceglie, M. G., Micheli, L., Muller, M. (2018). Quantifying soiling loss directly from PV yield. IEEE JPV 8(2)
  (soiling intervals between cleanings).
- Holmgren, W. F., Hansen, C. W., Mikofski, M. A. (2018). pvlib python. JOSS 3(29).
- Shapley, L. S. (1953). A value for n-person games.

MIT licence.
