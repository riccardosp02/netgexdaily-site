"""
Rilevamento pattern "P" (pole + flag) su dati OHLC.

Uso:
    python3 scripts/pole_flag_detect.py data/tuo_file.csv

CSV atteso con colonne (case-insensitive):
    datetime, open, high, low, close, volume

Dipendenze:
    pip install pandas numpy mplfinance
"""
import sys
import argparse
import numpy as np
import pandas as pd
import mplfinance as mpf


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def find_poles(df: pd.DataFrame, min_bars: int, max_bars: int, min_gain_pct: float):
    """Trova aste: movimenti rialzisti ripidi di ampiezza >= min_gain_pct su max_bars candele."""
    closes = df["close"].values
    poles = []
    n = len(closes)
    for start in range(n - min_bars):
        for length in range(min_bars, max_bars + 1):
            end = start + length
            if end >= n:
                break
            gain = (closes[end] - closes[start]) / closes[start] * 100
            if gain >= min_gain_pct:
                # richiede monotonia sommaria: niente grosse ritracciate nel mezzo
                segment = closes[start:end + 1]
                drawdown = (segment.max() - segment.min())
                if (segment[-1] - segment[0]) / drawdown > 0.6:
                    poles.append((start, end, gain))
    # tiene solo le aste più lunghe/non sovrapposte (greedy su gain decrescente)
    poles.sort(key=lambda p: -p[2])
    selected, used = [], np.zeros(n, dtype=bool)
    for s, e, g in poles:
        if not used[s:e + 1].any():
            selected.append((s, e, g))
            used[s:e + 1] = True
    selected.sort(key=lambda p: p[0])
    return selected


def find_flag_after_pole(df: pd.DataFrame, pole_end: int, min_bars: int, max_bars: int,
                          max_range_pct: float):
    """Cerca un box laterale (bassa volatilità) subito dopo la fine di un'asta."""
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    pole_high = high[pole_end]

    for length in range(min_bars, max_bars + 1):
        end = pole_end + length
        if end >= n:
            break
        segment_high = high[pole_end + 1:end + 1]
        segment_low = low[pole_end + 1:end + 1]
        if len(segment_high) == 0:
            continue
        box_top = segment_high.max()
        box_bottom = segment_low.min()
        range_pct = (box_top - box_bottom) / pole_high * 100
        if range_pct <= max_range_pct and box_top <= pole_high * 1.02:
            return pole_end + 1, end, box_top, box_bottom
    return None


def detect_pole_flag_patterns(df: pd.DataFrame,
                               pole_min_bars=3, pole_max_bars=15, pole_min_gain_pct=5.0,
                               flag_min_bars=3, flag_max_bars=20, flag_max_range_pct=4.0):
    patterns = []
    poles = find_poles(df, pole_min_bars, pole_max_bars, pole_min_gain_pct)
    for pole_start, pole_end, gain in poles:
        flag = find_flag_after_pole(df, pole_end, flag_min_bars, flag_max_bars, flag_max_range_pct)
        if flag:
            flag_start, flag_end, box_top, box_bottom = flag
            patterns.append({
                "pole_start": df.index[pole_start],
                "pole_end": df.index[pole_end],
                "pole_gain_pct": round(gain, 2),
                "flag_start": df.index[flag_start],
                "flag_end": df.index[flag_end],
                "box_top": box_top,
                "box_bottom": box_bottom,
            })
    return patterns


def plot_patterns(df: pd.DataFrame, patterns: list, out_path: str = None):
    addplots = []
    lines = []
    rects = []
    for p in patterns:
        lines.append([(p["pole_start"], df.loc[p["pole_start"], "low"]),
                      (p["pole_end"], df.loc[p["pole_end"], "high"])])
        rects.append((p["flag_start"], p["box_bottom"], p["flag_end"], p["box_top"]))

    fig, axlist = mpf.plot(
        df, type="candle", style="yahoo", returnfig=True,
        alines=dict(alines=lines, colors=["green"] * len(lines), linewidths=2),
        volume=False, figsize=(14, 7), title="Pole + Flag pattern detection",
    )
    ax = axlist[0]
    for x0, y0, x1, y1 in rects:
        ax.add_patch(
            __import__("matplotlib").patches.Rectangle(
                (mpf_date_to_num(x0), y0),
                mpf_date_to_num(x1) - mpf_date_to_num(x0),
                y1 - y0,
                fill=False, edgecolor="darkgreen", linewidth=2,
            )
        )
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Grafico salvato in {out_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def mpf_date_to_num(d):
    import matplotlib.dates as mdates
    return mdates.date2num(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default=None, help="Path immagine output (es. out.png)")
    parser.add_argument("--pole-min-gain", type=float, default=5.0)
    parser.add_argument("--flag-max-range", type=float, default=4.0)
    args = parser.parse_args()

    df = load_ohlc(args.csv_path)
    patterns = detect_pole_flag_patterns(
        df, pole_min_gain_pct=args.pole_min_gain, flag_max_range_pct=args.flag_max_range
    )

    if not patterns:
        print("Nessun pattern P (pole+flag) trovato con questi parametri.")
        sys.exit(0)

    print(f"Trovati {len(patterns)} pattern:")
    for p in patterns:
        print(f"  Asta: {p['pole_start']} -> {p['pole_end']} ({p['pole_gain_pct']}%)  "
              f"Box: {p['flag_start']} -> {p['flag_end']}  [{p['box_bottom']:.2f}-{p['box_top']:.2f}]")

    plot_patterns(df, patterns, out_path=args.out)


if __name__ == "__main__":
    main()
