#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Refresh-aware earners/players forecaster, stdlib only, no network.

Model, per trophy (all work in logit / log space):
  * Exophase refreshes its tracked-player count in batches. At a refresh the
    player count jumps and every rate is diluted: the new cohort holds far
    fewer trophies than the players tracked so far. So we forecast earners
    and players separately and take rate = 100 * earners / players.
  * Players: log-log fit of players_tracked on ln(1 + hours since release),
    extrapolated to the target. When history shows no refresh at all yet
    (launch night: flat 638 players) we fall back to the documented
    launch-phase prior of one refresh before the target with a ~6x player
    jump (the kit documents the first refresh 638 -> 3,868 players).
  * Earners: log-log fit of earners on ln(1 + hours since release) over the
    reads since the last refresh (the current regime only), extrapolated to
    the target; slope clamped to [0, 1.5] because earners never fall inside
    a regime. Expected new players between now and the target earn trophies
    at a cohort rate that follows a power law of the trophy's current rate:
    new_rate = 100 * (rate/100) ** gamma. Gamma is estimated from the
    refreshes actually observed in history (pooled log-log regression of
    new-cohort rate on pre-refresh rate); with no refresh observed yet the
    launch-night prior gamma = 1.57 is used (calibrated on the documented
    first refresh: 3,868 players, 4,276 trophies earned).
  * Spread: calibrated from backtests (see README). A base logit sigma,
    plus the in-sample fit residual scaled by horizon, plus a
    refresh-residual term: the per-refresh surprise that remains AFTER the
    dilution is modelled in the median (the raw jump is not the risk — the
    modelled dilution already accounts for it), scaled by the expected
    number of refreshes before the target. Spread is per class (rare
    trophies wider) and asymmetric: upside wider for fast-growing
    early-story trophies, downside wider for grind trophies.

Reads every *.in in $MAKEFUN_INPUT_DIR (default /input), writes one
*.out per case to $MAKEFUN_OUTPUT_DIR (default /output). Deterministic:
no randomness, no wall-clock use, fixed iteration order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- backtest-calibrated constants (see README / backtest notes) ---
GAMMA_PRIOR = 1.57        # new-cohort earn-rate power law, no refresh seen
PLAYER_JUMP_PRIOR = 6.0   # players multiplier of one launch-phase refresh
REFRESH_GAP_PRIOR_MIN = 300.0  # expected minutes between refreshes
JUMP_RESID_PRIOR = 0.10   # residual |logit| surprise per refresh after the
                          # dilution is modelled (backtest: median residual
                          # sd 0.05-0.09 logit on the practice refresh)
BASE_SIGMA = 0.10         # base logit sigma (rounding + shape risk)
SIGMA_FLOOR = 0.12        # never narrower than this (logit)
SIGMA_CAP = 1.40          # never wider than this (logit)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

TIER_PRIOR_RATE = {"platinum": 0.10, "gold": 0.20, "silver": 0.50, "bronze": 1.00}


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_rate(r):
    if r is None or (isinstance(r, float) and (math.isnan(r) or math.isinf(r))):
        return None
    return min(99.99, max(0.01, float(r)))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        e = math.exp(-x)
        return 100.0 / (1.0 + e)
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def round4_half_up(x):
    """Like JavaScript Math.round(x * 10000) / 10000 (x >= 0 here)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def fmt_num(x):
    """JSON number: integers without '.0', otherwise up to 4 decimals."""
    v = round4_half_up(x)
    if v == int(v):
        return str(int(v))
    s = ("%.4f" % v).rstrip("0").rstrip(".")
    return s if s else "0"


def is_story(desc):
    d = desc.lower()
    return "chapter" in d or "final mission" in d or "complete gone fishing" in d


def is_grind(desc):
    d = desc.lower()
    if any(ch.isdigit() for ch in d):
        for kw in ("complete", "win ", "rob ", "discover", "earn ", "spend",
                   "purchase", "register", "meditate", "hunt"):
            if kw in d:
                return True
    return False


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(horizon_min / 1440.0, 0.0)

    # ---- per-read series ----
    times_h = []      # hours since release
    players = []      # players_tracked per read
    read_rows = []    # (t_hours, players, rates) aligned with times_h/players
    for h in history:
        try:
            t = (parse_time(h["at"]) - RELEASE).total_seconds() / 3600.0
            p = float(h["players_tracked"])
        except (KeyError, TypeError, ValueError):
            continue
        if p <= 0:
            continue
        times_h.append(t)
        players.append(p)
        read_rows.append((t, p, h.get("rates") or []))
    n_reads = len(times_h)

    # ---- refresh detection (player-count jumps) ----
    refresh_idx = []
    for j in range(1, n_reads):
        if players[j] > players[j - 1] * 1.02:
            refresh_idx.append(j)
    n_refresh = len(refresh_idx)
    last_refresh = refresh_idx[-1] if refresh_idx else 0

    # mean gap between refreshes (minutes), for expected future refreshes
    if n_refresh >= 2:
        gaps = []
        prev_t = times_h[refresh_idx[0]]
        for j in refresh_idx[1:]:
            gaps.append((times_h[j] - prev_t) * 60.0)
            prev_t = times_h[j]
        mean_gap = sum(gaps) / len(gaps) if gaps else REFRESH_GAP_PRIOR_MIN
        mean_gap = min(max(mean_gap, 30.0), 1440.0)
    else:
        mean_gap = REFRESH_GAP_PRIOR_MIN
    n_future_refresh = horizon_min / mean_gap if mean_gap > 0 else 0.0
    n_future_refresh = min(max(n_future_refresh, 0.0), 24.0)
    if n_refresh == 0 and horizon_min >= 60:
        # No refresh seen yet but the horizon spans one: expect it.
        n_future_refresh = max(n_future_refresh, 1.0)

    # ---- players forecast: log-log fit ----
    p_target = None
    if n_reads >= 2:
        xs = [math.log1p(max(t, 0.0)) for t in times_h]
        ys = [math.log(p) for p in players]
        ws = [0.5 ** (max(times_h[-1] - t, 0.0) / 24.0) for t in times_h]
        fit = wls(xs, ys, ws)
        if fit is not None:
            a, b = fit
            b = min(max(b, 0.0), 2.0)
            x_t = math.log1p(max((target - RELEASE).total_seconds() / 3600.0, 0.0))
            p_target = math.exp(a + b * x_t)
    if p_target is None or p_target <= 0:
        p_target = players[-1] if players else 638.0
    p_last = players[-1] if players else p_target
    if n_refresh == 0 and horizon_min >= 60:
        # Launch-phase prior: one refresh, ~6x players, before the target.
        p_target = max(p_target, p_last * PLAYER_JUMP_PRIOR)
    # A log-log fit on a steep launch can extrapolate absurdly far over a
    # multi-day horizon; cap the implied growth at 100x the last count.
    p_target = min(max(p_target, p_last), p_last * 100.0)

    # ---- gamma (new-cohort power law) from observed refreshes ----
    gammas = []
    if n_refresh:
        for j in refresh_idx:
            if j == 0:
                continue
            p_before, p_after = players[j - 1], players[j]
            new_p = p_after - p_before
            if new_p <= 0:
                continue
            rates_before = read_rows[j - 1][2]
            rates_after = read_rows[j][2]
            xs_g, ys_g = [], []
            for k in range(min(len(rates_before), len(rates_after))):
                rb, ra = rates_before[k], rates_after[k]
                if rb is None or ra is None:
                    continue
                rb = clamp_rate(rb)
                ra = clamp_rate(ra)
                if rb is None or ra is None or rb >= 99.5:
                    continue
                e_before = rb / 100.0 * p_before
                e_after = ra / 100.0 * p_after
                r_new = (e_after - e_before) / new_p * 100.0
                if r_new <= 0.005 or r_new >= rb:
                    continue
                xs_g.append(math.log(rb / 100.0))
                ys_g.append(math.log(r_new / 100.0))
            if len(xs_g) >= 5:
                fit = wls(xs_g, ys_g, [1.0] * len(xs_g))
                if fit is not None and fit[1] > 0:
                    gammas.append(fit[1])
    if gammas:
        gammas.sort()
        gamma = gammas[len(gammas) // 2]
        gamma = min(max(gamma, 1.0), 2.0)
    else:
        gamma = GAMMA_PRIOR
    # Residual refresh surprise: stable gamma across refreshes means the
    # dilution is predictable; a drifting gamma means it is not.
    if len(gammas) >= 2:
        g_spread = min(max(gammas[-1] - gammas[0], 0.0), 1.0)
        jump_resid = JUMP_RESID_PRIOR + 0.5 * g_spread
    else:
        jump_resid = JUMP_RESID_PRIOR

    x_target = math.log1p(max((target - RELEASE).total_seconds() / 3600.0, 0.0))
    new_players = max(0.0, p_target - p_last)

    rows = []
    for ti, tr in enumerate(trophies):
        tier = tr.get("tier") or "bronze"
        desc = tr.get("description") or ""
        # series of (t, rate, players) for this trophy
        series = []
        for (t, p, rates) in read_rows:
            if ti >= len(rates):
                continue
            r = rates[ti]
            if r is None:
                continue
            r = clamp_rate(r)
            if r is None:
                continue
            series.append((t, r, p))
        last_rate = series[-1][1] if series else None

        if last_rate is None:
            # No history at all for this trophy: tier prior, wide spread.
            median = TIER_PRIOR_RATE.get(tier, 1.0)
            sigma = 1.0
            up_m, down_m = 1.0, 1.0
        else:
            # ---- earners fit on the current regime (since last refresh) ----
            seg = [(t, r, p) for (t, r, p) in series if t >= times_h[last_refresh]] \
                if n_reads else series
            if len(seg) < 3:
                seg = series
            e_base = None
            resid_sd = 0.10
            if len(seg) >= 2:
                xs = [math.log1p(max(t, 0.0)) for (t, r, p) in seg]
                ys = [math.log(max(r / 100.0 * p, 1e-9)) for (t, r, p) in seg]
                ws = [0.5 ** (max(seg[-1][0] - t, 0.0) / 24.0) for (t, r, p) in seg]
                fit = wls(xs, ys, ws)
                if fit is not None:
                    a, b = fit
                    b = min(max(b, 0.0), 1.5)
                    e_base = math.exp(a + b * x_target)
                    res = [y - (a + b * x) for x, y in zip(xs, ys)]
                    if len(res) >= 3:
                        m = sum(res) / len(res)
                        resid_sd = math.sqrt(sum((v - m) ** 2 for v in res) / len(res))
                        resid_sd = min(max(resid_sd, 0.05), 0.6)
            if e_base is None:
                e_base = last_rate / 100.0 * p_last
            # ---- new-cohort earners ----
            r_new = 100.0 * (last_rate / 100.0) ** gamma
            e_new = new_players * r_new / 100.0
            e_target = e_base + e_new
            median = 100.0 * e_target / p_target
            median = min(max(median, 0.01), 99.99)
            # ---- spread ----
            sigma_fit = resid_sd * math.sqrt(1.0 + horizon_days)
            sigma_ref = jump_resid * math.sqrt(n_future_refresh)
            sigma = math.sqrt(BASE_SIGMA ** 2 + sigma_fit ** 2 + sigma_ref ** 2)
            # per-class width: rare trophies are noisier in logit space
            if last_rate < 1.0:
                sigma *= 1.30
            elif last_rate > 10.0:
                sigma *= 0.85
            if tier in ("gold", "platinum"):
                sigma *= 1.15
            sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
            # asymmetric: upside wider for fast story trophies, downside
            # wider for grind trophies
            if is_story(desc):
                up_m, down_m = 1.25, 1.0
            elif is_grind(desc):
                up_m, down_m = 1.0, 1.25
            else:
                up_m, down_m = 1.1, 1.1

        m_logit = logit(median)
        qs = []
        for z in Z:
            mult = up_m if z > 0 else (down_m if z < 0 else 1.0)
            q = inv_logit(m_logit + z * sigma * mult)
            qs.append(min(max(q, 0.01), 99.99))
        # round half-up to 4 decimals, keep non-decreasing
        qs = [round4_half_up(q) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append((tr["id"], qs))

    out = {"trophies": [{"id": tid, "q": qs} for tid, qs in rows]}
    return out


def dump_out(obj):
    parts = ['{"trophies":[']
    first = True
    for row in obj["trophies"]:
        if not first:
            parts.append(",")
        first = False
        qs = ",".join(fmt_num(q) for q in row["q"])
        parts.append('{"id":%d,"q":[%s]}' % (row["id"], qs))
    parts.append("]}")
    return "".join(parts)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        text = dump_out(out)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
