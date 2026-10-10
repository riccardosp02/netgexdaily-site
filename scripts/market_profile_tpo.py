"""
Market Profile in stile TPO (Time Price Opportunity), come l'indicatore
"Market Profile" di ATAS (diverso dal Volume Profile: qui si conta il
TEMPO passato a ogni livello, non il volume scambiato).

Metodo:
    - la sessione e' divisa in periodi fissi (default 30 min), uno per
      lettera (A, B, C, ...)
    - per ogni barra 5m, ogni livello di prezzo toccato (tra low e high,
      arrotondato al tick) riceve la lettera del periodo corrente
    - POC = livello con piu' lettere diverse (dove il prezzo e' tornato
      piu' spesso in periodi diversi)
    - Value Area = fascia che copre ~70% delle lettere totali, espansa
      dal POC verso il bin adiacente piu' popolato
    - Initial Balance = range dei primi 2 periodi (prima ora)

Uso:
    python3 scripts/market_profile_tpo.py data/ohlc_cvd.csv --start "2026-10-01 00:00:00" \
        --end "2026-10-01 23:55:00" --out profilo.png

Dipendenze:
    pip install pandas numpy matplotlib mplfinance
"""
import argparse
import string
import numpy as np
import pandas as pd
import mplfinance as mpf

from phase_detect import load_data, resample_ohlc


def build_tpo_profile(df_5m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                       tick: float = 0.5, period_min: int = 30, va_coverage: float = 0.70):
    day = df_5m.loc[start:end].copy()
    letters = string.ascii_uppercase + string.ascii_lowercase
    day["period"] = ((day.index - day.index[0]).total_seconds() // (period_min * 60)).astype(int)

    tpo = {}
    for _, row in day.iterrows():
        lo = np.floor(row["low"] / tick) * tick
        hi = np.ceil(row["high"] / tick) * tick
        letter = letters[row["period"] % len(letters)]
        lvl = lo
        while lvl <= hi:
            tpo.setdefault(round(lvl, 2), set()).add(letter)
            lvl += tick

    levels = sorted(tpo.keys())
    counts = {lvl: len(tpo[lvl]) for lvl in levels}
    poc = max(counts, key=counts.get)
    total = sum(counts.values())

    poc_idx = levels.index(poc)
    lo_i, hi_i = poc_idx, poc_idx
    covered = counts[poc]
    while covered / total < va_coverage and (lo_i > 0 or hi_i < len(levels) - 1):
        left = counts[levels[lo_i - 1]] if lo_i > 0 else -1
        right = counts[levels[hi_i + 1]] if hi_i < len(levels) - 1 else -1
        if left >= right and lo_i > 0:
            lo_i -= 1
            covered += counts[levels[lo_i]]
        elif hi_i < len(levels) - 1:
            hi_i += 1
            covered += counts[levels[hi_i]]
        else:
            break
    val, vah = levels[lo_i], levels[hi_i]

    ib_bars = day[day["period"] <= 1]
    ib_low, ib_high = ib_bars["low"].min(), ib_bars["high"].max()

    return {
        "tpo": tpo, "levels": levels, "counts": counts,
        "poc": poc, "val": val, "vah": vah,
        "va_coverage": covered / total,
        "ib_low": ib_low, "ib_high": ib_high,
    }


def plot_profile(df_5m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, profile: dict,
                  timeframe: str = "15min", out_path: str = None):
    df_tf = resample_ohlc(df_5m, timeframe) if timeframe != "5min" else df_5m
    ohlc = df_tf.loc[start:end, ["open", "high", "low", "close"]].copy()
    ohlc.index.name = "Date"
    ohlc.columns = ["Open", "High", "Low", "Close"]

    fig, axlist = mpf.plot(
        ohlc, type="candle", style="yahoo", returnfig=True, figsize=(18, 9), volume=False,
        title=f"{start.date()} - Market Profile (TPO) - POC/VA/IB",
    )
    ax = axlist[0]
    ax.axhline(profile["poc"], color="red", linewidth=1.5, label=f"POC {profile['poc']:.2f}")
    ax.axhline(profile["val"], color="purple", linewidth=1, linestyle="--", label=f"VAL {profile['val']:.2f}")
    ax.axhline(profile["vah"], color="purple", linewidth=1, linestyle="--", label=f"VAH {profile['vah']:.2f}")
    ax.axhspan(profile["val"], profile["vah"], color="purple", alpha=0.08)
    ax.axhline(profile["ib_low"], xmax=0.08, color="blue", linewidth=1)
    ax.axhline(profile["ib_high"], xmax=0.08, color="blue", linewidth=1)
    ax.text(1, profile["ib_high"] + 1,
            f"IB ({profile['ib_low']:.0f}-{profile['ib_high']:.0f})", color="blue", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in {out_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tick", type=float, default=0.5, help="Ampiezza riga di prezzo (TPO block)")
    parser.add_argument("--period-min", type=int, default=30, help="Minuti per periodo/lettera")
    parser.add_argument("--va-coverage", type=float, default=0.70)
    parser.add_argument("--timeframe", default="15min", help="Timeframe candele nel grafico")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df_5m = load_data(args.csv_path)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    profile = build_tpo_profile(df_5m, start, end, tick=args.tick,
                                 period_min=args.period_min, va_coverage=args.va_coverage)

    print(f"POC: {profile['poc']:.2f}")
    print(f"Value Area: {profile['val']:.2f} - {profile['vah']:.2f}  "
          f"(copertura {profile['va_coverage']*100:.0f}%)")
    print(f"Initial Balance: {profile['ib_low']:.2f} - {profile['ib_high']:.2f}")

    plot_profile(df_5m, start, end, profile, timeframe=args.timeframe, out_path=args.out)


if __name__ == "__main__":
    main()
