"""
Confronto visivo tra due metodi di rilevamento fasi/assorbimenti/breakout:
  A) box 30m accumulato (classify_phases + build_running_edges)
  B) Value Area (VAH/VAL, aggiornata ogni 5m, piu' stretta e reattiva)

Uso:
    python3 scripts/compare_30m_va.py data/ohlc_cvd.csv data/value_area.csv \
        --start "2026-10-07 00:00:00" --end "2026-10-07 09:00:00" --out confronto.png

Dipendenze:
    pip install pandas numpy matplotlib
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from phase_detect import (
    load_data, load_value_area, resample_30m, classify_phases,
    apply_min_duration, build_running_edges, refine_signal_start, detect_va_events,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV OHLC+CVD (date,time,open,high,low,close,cvd,vwap)")
    parser.add_argument("va_path", help="CSV Value Area (date,time,price,vah,val,volume)")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--impulse-z", type=float, default=1.3)
    parser.add_argument("--lateral-z", type=float, default=0.5)
    parser.add_argument("--confirm-bars", type=int, default=1)
    parser.add_argument("--min-phase-bars", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df_5m = load_data(args.csv_path)
    va = load_value_area(args.va_path)
    ohlc_30m = resample_30m(df_5m)
    result = classify_phases(ohlc_30m, args.lookback, args.impulse_z, args.lateral_z, args.confirm_bars)
    result = apply_min_duration(result, args.min_phase_bars)
    result = build_running_edges(result)

    refined = refine_signal_start(df_5m, result)
    va_events = detect_va_events(df_5m, va, result)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    price = df_5m.loc[start:end, "close"]
    box = result.loc[start:end]
    va_w = va.loc[start:end]
    refined_w = refined[(refined["bar_30m_end"] >= start) & (refined["bar_30m_end"] <= end)]
    va_events_w = va_events[(va_events["datetime"] >= start) & (va_events["datetime"] <= end)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax1.plot(price.index, price.values, color="black", linewidth=1, label="Close 5m")
    in_impulso = box["phase"] == "impulso"
    ax1.fill_between(box.index, price.min(), price.max(), where=in_impulso,
                      color="orange", alpha=0.15, step="post", label="Impulso (30m)")
    ax1.step(box.index, box["edge_upper"], where="post", color="green", linestyle="--",
              linewidth=1.2, label="Bordo box 30m")
    ax1.step(box.index, box["edge_lower"], where="post", color="green", linestyle="--", linewidth=1.2)
    for i, (_, r) in enumerate(refined_w.iterrows()):
        ax1.scatter(r["pivot_time"], r["pivot_close"], marker="v" if r["direction"] == "down" else "^",
                    color="red", s=90, zorder=5, label="Assorbimento (30m)" if i == 0 else "")
        if pd.notna(r["breakout_time"]):
            ax1.scatter(r["breakout_time"], r["breakout_close"], marker="x", color="blue", s=90,
                        zorder=5, label="Breakout conf. (30m)" if i == 0 else "")
    ax1.set_title("Metodo A: box 30m accumulato (fasi lateralita'/impulso)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2)

    ax2.plot(price.index, price.values, color="black", linewidth=1, label="Close 5m")
    ax2.plot(va_w.index, va_w["vah"], color="purple", linewidth=1, label="VAH")
    ax2.plot(va_w.index, va_w["val"], color="teal", linewidth=1, label="VAL")
    ax2.fill_between(va_w.index, va_w["val"], va_w["vah"], color="purple", alpha=0.08)
    absorb = va_events_w[va_events_w["type"] == "assorbimento"]
    breakout = va_events_w[va_events_w["type"] == "breakout"]
    ax2.scatter(absorb["datetime"], absorb["extreme"], marker="v", color="red", s=70,
                zorder=5, label="Assorbimento (VA)")
    ax2.scatter(breakout["datetime"], breakout["close"], marker="x", color="blue", s=70,
                zorder=5, label="Breakout (VA)")
    ax2.set_title("Metodo B: Value Area (VAH/VAL, aggiornata ogni 5m)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle(f"Confronto metodi — {start.date()}, {start.strftime('%H:%M')}-{end.strftime('%H:%M')}",
                 fontsize=13)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
