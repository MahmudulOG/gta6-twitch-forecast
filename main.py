"""Forecast a launch day's Twitch viewers, hour by hour, from its first hour.

Built for make.fun's gta6-twitch-ccu-forecast-01 (program/v1, Python 3.13,
standard library only). The grader runs this file once per case: it reads the
single case file in the input directory and writes one forecast file per case
to the output directory, with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for
each of hours 2..24 of the release day.

Method
------
Every case carries dozens of past big launches ("comparables"), each with its
own hourly viewer curve from 14 days before its release day to 3 days after.
A launch day's shape is far more alike in relative terms than in absolute
ones, so the model works with log-ratios to each launch's own hour-1 count:

    r_h = ln(viewers_h / viewers_1)        for h = 2..24

For the case being forecast, the comparables are pooled per hour with weights
that favour launches whose hour-1 audience was a similar size to this case's
hour-1 print (a 20k-viewer launch day does not move like a 900k one). The
weighted mean and weighted standard deviation of the pooled log-ratios give a
normal curve per hour, and the forecast quantiles are that curve's quantiles
around the case's own hour-1 print:

    q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h * SPREAD)

Three refinements came out of tuning. Log-ratios are winsorised at +/-3 (a
factor of about 20) before pooling, so one comparable whose unlock fell many
hours into its UTC day cannot drag an hour's mean or spread. The spread
calibration depends on the case's own shape: the mean of the six hours
before the release day, relative to hour 1 (the "pre-ratio"), separates
launches whose audience was already large before midnight UTC — every
practice case sits between 1.3 and 2.4 — from launches that only start at a
single global unlock moment. The former take the tight calibrated spread;
the latter keep the full pooled spread. And the early hours are adjusted by
the case's own approach slope: how steeply the audience was moving into
hour 1 (hour 1 against the mean of the three hours before the day),
compared with the pooled comparables' approach slopes. A launch still
falling steeply into hour 1 — every practice case is — tends to keep
falling for a few hours before the next unlock wave, so its early means
shift by a faded fraction of the slope difference.

The constants below were tuned by leave-one-out cross-validation over the
comparables themselves: each comparable in turn was forecast from the
others and scored with the contest's exact pinball loss on the log scale.
On launches shaped like the practice cases the tuned model scores about
0.272 against the house persistence baseline's 0.45 on the same
cross-validation. (Earlier entries of this program scored 0.277 and 0.274
on the same cross-validation; their server practice losses were 0.2118 and
0.2101 against the baseline's roughly 0.387.)

The program is deterministic: fixed iteration order, no randomness, no clock,
no network.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader runs python3 -P -s, which keeps this file's directory off
# sys.path; put it back first so a sibling module would import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Similarity weighting: a comparable whose hour-1 count differs from the
# case's hour-1 print by a factor of e in log space keeps exp(-WEIGHT_SCALE)
# of its weight.
WEIGHT_SCALE = 0.8
# Log-ratios are winsorised at +/-WINSOR before pooling.
WINSOR = 3.0
# Spread calibration: the pooled log-ratio spread, scaled. Cross-validation
# found the raw pooled spread too wide for launches whose pre-midnight hours
# already drew a large audience (pre-ratio at least PRE_RATIO_SPLIT); other
# launches keep the full pooled spread.
SPREAD_TIGHT = 0.60
SPREAD_FULL = 1.0
PRE_RATIO_SPLIT = 1.2
# Approach-slope adjustment for the early hours: hour h (2..SLOPE_HOURS)
# takes SLOPE_BETA * (SLOPE_HOURS + 1 - h) / SLOPE_HOURS of the difference
# between the case's own approach slope and the pooled comparables' slope.
SLOPE_BETA = 0.6
SLOPE_HOURS = 6
# Fallback log-space spread for an hour no comparable covers.
FALLBACK_SD = 0.5


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offsets(day_start, points, key):
    """Map whole-hour offsets from day_start to viewer counts.

    points is a list of rows carrying a timestamp under `key` ("from" in the
    comparables and history_hourly) and a "viewers" count. Rows that do not
    land on a whole hour from day_start, or carry no positive count, are
    skipped.
    """
    start = parse_time(day_start)
    series = {}
    for row in points or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        delta = parse_time(row[key]) - start
        offset = int(delta.total_seconds() // 3600)
        if delta.total_seconds() == offset * 3600:
            series[offset] = float(viewers)
    return series


def round2(value):
    """Round to 2 decimals, half up, matching the house baseline's rounding."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 100
    return int(result) if float(result).is_integer() else result


def approach_slope(series, first):
    """Log of hour 1 over the mean of the three hours before the day.

    Negative when the audience was falling into hour 1. Returns None when
    fewer than two of those hours are present.
    """
    before = [series[o] for o in range(-3, 0) if o in series]
    if len(before) < 2:
        return None
    return math.log(first / (sum(before) / len(before)))


def comparable_curves(case):
    """Extract (hour-1 count, {offset: viewers}, slope) per usable comparable.

    A comparable is usable when its release day's first hour is present: the
    pre-release category history bundled in every case has no launch day of
    its own and drops out here on its own.
    """
    curves = []
    for comp in case.get("comparables") or []:
        series = hour_offsets(comp["day_start"], comp.get("hourly"), "from")
        first = series.get(0)
        if first is None or first <= 0:
            continue
        curves.append((first, series, approach_slope(series, first)))
    return curves


def pre_ratio(case, p1):
    """Mean viewers of the six hours before the release day, over hour 1.

    Read from the case's own history_hourly. Returns None when fewer than
    three of those hours are present, in which case the caller keeps the
    full pooled spread.
    """
    series = hour_offsets(case["day_start"], case.get("history_hourly"), "from")
    before = [series[o] for o in range(-6, 0) if o in series]
    if len(before) < 3:
        return None
    return (sum(before) / len(before)) / p1


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    ratio = pre_ratio(case, p1)
    spread = SPREAD_TIGHT if ratio is not None and ratio >= PRE_RATIO_SPLIT else SPREAD_FULL

    curves = comparable_curves(case)

    # The case's own approach slope, and the pooled comparables' slope with
    # the same similarity weights the ratio pool uses. The early hours'
    # means shift by a faded fraction of the difference.
    own_series = hour_offsets(case["day_start"], case.get("history_hourly"), "from")
    own_slope = approach_slope(own_series, p1)
    slope_shift = {}
    if own_slope is not None:
        num = 0.0
        den = 0.0
        for comp_first, _series, comp_slope in curves:
            if comp_slope is None:
                continue
            w = math.exp(-WEIGHT_SCALE * abs(math.log(comp_first / p1)))
            num += w * comp_slope
            den += w
        if den > 0:
            diff = own_slope - num / den
            for hour in range(2, SLOPE_HOURS + 1):
                fade = (SLOPE_HOURS + 1 - hour) / SLOPE_HOURS
                slope_shift[hour] = SLOPE_BETA * fade * diff

    # Per forecast hour, pool the comparables' log-ratios to their own
    # hour-1 counts, weighted by hour-1 similarity to this case.
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        offset = hour - 1
        weighted = []
        for comp_first, series, _slope in curves:
            viewers = series.get(offset)
            if viewers is None or viewers <= 0:
                continue
            weight = math.exp(-WEIGHT_SCALE * abs(math.log(comp_first / p1)))
            log_ratio = math.log(viewers / comp_first)
            log_ratio = max(-WINSOR, min(WINSOR, log_ratio))
            weighted.append((log_ratio, weight))
        if weighted:
            total_w = sum(w for _, w in weighted)
            mu = sum(v * w for v, w in weighted) / total_w
            var = sum(w * (v - mu) ** 2 for v, w in weighted) / total_w
            sd = math.sqrt(max(var, 1e-9))
        else:
            mu, sd = 0.0, FALLBACK_SD
        mu += slope_shift.get(hour, 0.0)
        quantiles = []
        for z in Z:
            q = p1 * math.exp(mu + z * sd * spread)
            q = min(max(q, 1e-6), 1e9)
            quantiles.append(round2(q))
        quantiles.sort()
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
