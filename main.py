#!/usr/bin/env python3
"""GTA VI launch-day Twitch CCU forecaster (make.fun program/v1).

Model: per-hour log-linear regression on the hour-1 print.
-----------------------------------------------------------
For every launch in the case file's comparables we know the full release
day: the hour-1 print h1 and the viewers v_h at each later hour h.
Across ~40 big game launches the relationship between log(v_h) and
log(h1) is close to linear per hour, with a slope that falls through the
day: ~1.0 in the small hours (viewers track h1 almost 1:1 while the
world sleeps), ~0.7 by hour 9, ~0.45 by hour 12, and ~0.05-0.1 in the
evening (evening viewership is only weakly tied to the hour-1 print --
it is driven by the US unlocks and the European evening, which had not
happened yet at hour 1).  So for each hour 2..24 we fit

    log(v_h) = a_h + b_h * log(h1) + eps

by ordinary least squares over the comparables, and take the forecast
quantiles from the residual distribution: per-hour empirical residual
quantiles, shrunk toward the pooled empirical quantiles (k=10
pseudo-counts -- n_h ~ 37 points per hour is too few for the 5%/95%
tails on their own), blended 0.75/0.25 with normal quantiles at the
per-hour residual sd.  The residuals are winsorized first at their
pooled 2.5%/97.5% quantiles: a few launches have extreme evening
residuals (10x nights), and without the clip those outliers stretch
every hour's empirical quantiles.  Evening residuals are wider and
more skewed than small-hour ones, so the hour-specific shape matters.
That gives honest, wide intervals -- the residual sd is ~0.3 log units
in the small hours and ~0.8-1.1 in the evening, i.e. the 5-95% band
spans roughly a factor 10 at the peak -- because the evening level of
a launch is genuinely hard to pin down from hour 1 alone (across
launches it ranges from below h1 to >10x h1).

Validation (leave-one-out over the 38 practice comparables, and the
exact Wolverine practice answers):
  * this model:  LOO mean pinball 0.435, median 0.341; Wolverine 0.204
    (before the quantile fix: LOO 0.438/0.350, Wolverine 0.205 -- the
    fix improves all four: LOO mean, median, big-launch subset
    0.349 -> 0.341, and Wolverine)
  * house baseline (flat at h1, spread 0.15*sqrt(h-1)):
                 LOO mean pinball 0.930, median 0.325; Wolverine 0.744
  * median-ratio curve: LOO 0.803; k-NN on the pre-midnight trajectory:
    LOO 0.66-0.74.  Adding pre-midnight features (level at -1h, ramp
    slope, pre-midnight peak) did not improve LOO -- with ~40 launches
    the extra coefficients only add variance.  Also tried and rejected:
    smoothing the OLS slope across hours (LOO 0.444), Student-t tails
    (0.439), shrinking the per-hour sd toward the global sd (0.439),
    and winsorizing beyond 5% (LOO mean turns back up at 7.5%).

Deterministic (no randomness), stdlib only, runs in well under a second.
Reads every *.in file in the input dir, writes one *.out per case.
"""
import json
import math
import os
import statistics
import sys

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# standard normal quantiles for LEVELS
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
HOURS = range(2, 25)
MAX_Q = 1_000_000_000.0


def parse_ts(s):
    """'2026-09-15T00:00:00Z' -> epoch hours (float)."""
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp() / 3600.0


def series_offsets(hourly, day_start):
    """hourly: list of {from, viewers} -> {offset_hours: viewers}."""
    base = parse_ts(day_start)
    out = {}
    for p in hourly or []:
        try:
            off = int(round(parse_ts(p["from"]) - base))
        except Exception:
            continue
        v = p.get("viewers")
        if v is not None:
            out[off] = v
    return out


def ols(X, y):
    """Least squares y ~ X (rows include intercept). Gaussian elimination."""
    n = len(X)
    p = len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)]
         + [sum(X[k][i] * y[k] for k in range(n))] for i in range(p)]
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        if abs(A[i][i]) < 1e-12:
            continue
        for r in range(i + 1, p):
            f = A[r][i] / A[i][i]
            for c in range(i, p + 1):
                A[r][c] -= f * A[i][c]
    beta = [0.0] * p
    for i in reversed(range(p)):
        s = A[i][p] - sum(A[i][j] * beta[j] for j in range(i + 1, p))
        beta[i] = s / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0
    return beta


def baseline(h1):
    """House baseline fallback: flat at h1, spread 0.15*sqrt(h-1)."""
    out = {}
    for h in HOURS:
        s = 0.15 * math.sqrt(h - 1)
        qs = [h1 * math.exp(z * s) for z in Z]
        out[h] = [max(1.0, min(MAX_Q, q)) for q in qs]
    return out


def forecast(h1, comparables):
    """Return {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    pool = []
    for comp in comparables or []:
        try:
            s = series_offsets(comp.get("hourly"), comp.get("day_start"))
        except Exception:
            continue
        if s.get(0):
            pool.append(s)
    if len(pool) < 8 or not h1 or h1 <= 0:
        return baseline(h1 if h1 and h1 > 0 else 1000.0)

    lh1 = math.log(h1)
    all_resid = []
    per_hour = {}
    for h in HOURS:
        X, y = [], []
        for s in pool:
            v = s.get(h - 1)
            h1c = s.get(0)
            if v and v > 0 and h1c and h1c > 0:
                X.append([1.0, math.log(h1c)])
                y.append(math.log(v))
        if len(X) < 8:
            per_hour[h] = None
            continue
        beta = ols(X, y)
        resid = [yy - (beta[0] + beta[1] * xx[1]) for xx, yy in zip(X, y)]
        all_resid.extend(resid)
        per_hour[h] = (beta, resid)

    if not all_resid:
        return baseline(h1)

    all_resid.sort()
    n = len(all_resid)

    def rq(vals, tau):
        i = tau * (len(vals) - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return vals[lo] + (vals[hi] - vals[lo]) * (i - lo)

    # Winsorize: a few launches have extreme evening residuals (10x nights);
    # clipping the residuals at their pooled 2.5%/97.5% quantiles keeps those
    # outliers from stretching every hour's empirical quantiles. The clip is
    # applied to the pooled list and to each hour's residuals alike.
    lo_w, hi_w = rq(all_resid, 0.025), rq(all_resid, 0.975)
    all_resid = [min(max(r, lo_w), hi_w) for r in all_resid]
    for h in per_hour:
        if per_hour[h] is not None:
            b, rs = per_hour[h]
            per_hour[h] = (b, [min(max(r, lo_w), hi_w) for r in rs])

    rqs = [rq(all_resid, t) for t in LEVELS]
    gsd = statistics.pstdev(all_resid) if n > 1 else 0.5

    out = {}
    for h in HOURS:
        ph = per_hour[h]
        if ph is None:
            out[h] = baseline(h1)[h]
            continue
        beta, resid = ph
        mu = beta[0] + beta[1] * lh1
        sd_h = statistics.pstdev(resid) if len(resid) > 1 else gsd
        # Per-hour empirical residual quantiles, shrunk toward the pooled
        # ones (n_h ~ 37 points per hour is too few for the 5%/95% tails
        # on their own; k=10 pseudo-counts): evening residuals are wider
        # and more skewed than small-hour ones, so the hour-specific shape
        # matters. Blend 0.75 empirical / 0.25 normal.
        rs = sorted(resid)
        m = len(rs)
        rq_h = [rq(rs, t) for t in LEVELS]
        rq_use = [(m * a + 10.0 * b) / (m + 10.0) for a, b in zip(rq_h, rqs)]
        qs = []
        for j in range(5):
            q_emp = mu + rq_use[j]
            q_nor = mu + Z[j] * sd_h
            qs.append(math.exp(0.75 * q_emp + 0.25 * q_nor))
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        out[h] = [max(1.0, min(MAX_Q, q)) for q in qs]
    return out


def process_case(case):
    known = case.get("known") or []
    h1 = None
    if known:
        h1 = known[0].get("viewers")
    if not h1 or h1 <= 0:
        # fall back to the last pre-midnight hourly value
        hist = case.get("history_hourly") or []
        if hist:
            h1 = hist[-1].get("viewers")
    if not h1 or h1 <= 0:
        h1 = 1000.0
    fc = forecast(float(h1), case.get("comparables"))
    hours = []
    for h in HOURS:
        qs = [int(round(q)) for q in fc[h]]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        qs = [max(1, min(int(MAX_Q), q)) for q in qs]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not files:
        print(f"no .in files in {in_dir}", file=sys.stderr)
        return 1
    for fname in files:
        with open(os.path.join(in_dir, fname)) as f:
            case = json.load(f)
        result = process_case(case)
        out_name = fname[:-3] + ".out" if fname.endswith(".in") else fname + ".out"
        with open(os.path.join(out_dir, out_name), "w") as f:
            json.dump(result, f, separators=(",", ":"))
        print(f"{fname} -> {out_name}: h1={case.get('known', [{}])[0].get('viewers')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
