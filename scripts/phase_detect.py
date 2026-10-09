"""
Identificazione fase di mercato (lateralita' / impulso) su candele 30m,
usando le chiusure, man mano che il tempo avanza. Quando una fase e'
lateralita', cerca inoltre assorbimenti ai bordi del box sulle 5m,
confermati dal CVD.

Uso:
    python3 scripts/phase_detect.py data/ohlc_cvd_2026-10.txt --start "2026-10-02 09:00:00"

CSV atteso con colonne: date,time,open,high,low,close,cvd,vwap
(timeframe nativo 5m; le fasi vengono calcolate su chiusure 30m)

Dipendenze:
    pip install pandas numpy matplotlib
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.set_index("datetime").sort_index()
    return df


def resample_30m(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLC a 30 minuti. Il bin [T, T+30m) viene etichettato con end_time = T+30m,
    cioe' il momento in cui la barra 30m e' effettivamente chiusa e disponibile.
    """
    agg = df.resample("30min", label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    agg = agg.rename_axis("end_time")
    return agg


def classify_phases(ohlc_30m: pd.DataFrame, lookback: int, impulse_z: float,
                     lateral_z: float, confirm_bars: int) -> pd.DataFrame:
    """
    Per ogni barra 30m calcola la "velocita'" del prezzo (variazione della
    chiusura rispetto alla barra precedente) normalizzata sulla volatilita'
    recente (rolling std dei ritorni, lookback barre).

    Stato con isteresi:
      - entra in IMPULSO se la velocita' (z-score) resta sopra impulse_z per
        confirm_bars barre consecutive
      - torna in LATERALITA' se resta sotto lateral_z per confirm_bars barre
        consecutive
      - altrimenti mantiene la fase corrente (evita sfarfallio)
    """
    closes = ohlc_30m["close"]
    returns = closes.diff()
    vol = returns.rolling(lookback).std()
    z = (returns / vol).fillna(0)

    phases = []
    phase = "lateralita"  # stato iniziale prudente
    above_count = 0
    below_count = 0

    for val in z.abs():
        if val >= impulse_z:
            above_count += 1
            below_count = 0
        elif val <= lateral_z:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0

        if phase == "lateralita" and above_count >= confirm_bars:
            phase = "impulso"
        elif phase == "impulso" and below_count >= confirm_bars:
            phase = "lateralita"

        phases.append(phase)

    out = ohlc_30m.copy()
    out["z_velocity"] = z
    out["phase"] = phases
    return out


def apply_min_duration(result_30m: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    """
    Riassorbe nella fase precedente qualsiasi tratto piu' corto di min_bars
    barre da 30m (elimina blip isolati tipo una singola barra 'impulso'
    circondata da lateralita').
    """
    if min_bars <= 1:
        return result_30m

    phases = result_30m["phase"].tolist()
    changed = True
    while changed:
        changed = False
        seg_id = pd.Series(phases).ne(pd.Series(phases).shift()).cumsum()
        counts = seg_id.value_counts()
        for sid, length in counts.items():
            if length < min_bars and sid > 1:  # mai il primissimo tratto, non ha un "prima"
                idx = seg_id[seg_id == sid].index
                prev_phase = phases[idx[0] - 1]
                for i in idx:
                    phases[i] = prev_phase
                changed = True
                break  # ricalcola i segmenti dopo ogni merge

    out = result_30m.copy()
    out["phase"] = phases
    return out


def refine_signal_start(df_5m: pd.DataFrame, result_30m: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni transizione lateralita'->impulso vengono riportati due momenti,
    entrambi causali (nessun dato futuro rispetto al loro istante):

      - pivot_time / tipo "assorbimento": il massimo (per un impulso
        ribassista) o il minimo (per uno rialzista) toccato PRIMA che il
        movimento partisse davvero. E' il punto dove il mercato ha
        respinto un'aggressione, non ancora una rottura.
      - breakout_time / tipo "breakout confermato": la prima barra 5m la
        cui CHIUSURA e' gia' fuori dal bordo del box noto fino a quel
        momento (edge_upper/edge_lower della barra 30m precedente). E'
        la prima conferma che il prezzo non e' piu' dentro il range.
    """
    transitions = result_30m[
        (result_30m["phase"] == "impulso") & (result_30m["phase"].shift() == "lateralita")
    ]

    refined = []
    for end_time, row in transitions.iterrows():
        # guarda anche nella barra 30m precedente: il vero punto di svolta puo'
        # trovarsi li' se il movimento e' iniziato a ridosso del confine tra le due
        window_start = end_time - pd.Timedelta("60min")
        window = df_5m.loc[window_start:end_time]
        if window.empty:
            continue

        prev_30m_close = result_30m["close"].shift().loc[end_time]
        direction = "down" if row["close"] < prev_30m_close else "up"
        if direction == "down":
            pivot_time = window["close"].idxmax()
        else:
            pivot_time = window["close"].idxmin()
        pivot_close = window.loc[pivot_time, "close"]

        # bordo del box noto fino a quel momento: ultima barra 30m lateral
        # gia' chiusa prima del pivot (nessun dato oltre quell'istante)
        prior_closed = result_30m[result_30m.index <= pivot_time]
        breakout_time, breakout_close = None, None
        if not prior_closed.empty and prior_closed.iloc[-1]["phase"] == "lateralita" \
                and "edge_upper" in prior_closed.columns:
            edge_upper = prior_closed["edge_upper"].iloc[-1]
            edge_lower = prior_closed["edge_lower"].iloc[-1]
            after_pivot = window.loc[pivot_time:]
            if direction == "down" and not pd.isna(edge_lower):
                outside = after_pivot[after_pivot["close"] < edge_lower]
            elif direction == "up" and not pd.isna(edge_upper):
                outside = after_pivot[after_pivot["close"] > edge_upper]
            else:
                outside = after_pivot.iloc[0:0]
            if not outside.empty:
                breakout_time = outside.index[0]
                breakout_close = outside["close"].iloc[0]

        refined.append({
            "bar_30m_end": end_time, "direction": direction,
            "pivot_time": pivot_time, "pivot_close": pivot_close,
            "breakout_time": breakout_time, "breakout_close": breakout_close,
        })

    return pd.DataFrame(refined)


def build_running_edges(result_30m: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni barra 30m in fase 'lateralita', calcola il bordo superiore e
    inferiore del box come max(high)/min(low) accumulato dall'inizio del
    tratto laterale corrente (si allarga mano a mano che il tempo passa,
    senza guardare avanti). In fase 'impulso' non ci sono bordi.
    """
    edge_upper, edge_lower = [], []
    seg_high, seg_low = None, None
    prev_phase = None

    for _, row in result_30m.iterrows():
        if row["phase"] == "lateralita":
            if prev_phase != "lateralita":
                seg_high, seg_low = row["high"], row["low"]
            else:
                seg_high = max(seg_high, row["high"])
                seg_low = min(seg_low, row["low"])
            edge_upper.append(seg_high)
            edge_lower.append(seg_low)
        else:
            seg_high, seg_low = None, None
            edge_upper.append(np.nan)
            edge_lower.append(np.nan)
        prev_phase = row["phase"]

    out = result_30m.copy()
    out["edge_upper"] = edge_upper
    out["edge_lower"] = edge_lower
    return out


def detect_absorptions(df_5m: pd.DataFrame, edges_30m: pd.DataFrame, min_wick: float = 0.0) -> pd.DataFrame:
    """
    Per ogni barra 5m, usa i bordi del box noti dall'ULTIMA barra 30m gia'
    chiusa prima di quel momento (nessun look-ahead). Se la 30m di
    riferimento e' in fase laterale e la barra 5m ha una wick che esce dal
    bordo ma chiude di nuovo dentro il box, segnala un possibile
    assorbimento. Il delta di CVD della barra 5m conferma la direzione
    dell'aggressione respinta.
    """
    bars = df_5m.copy()
    bars["cvd_delta"] = bars["cvd"].diff()

    ref = edges_30m[["phase", "edge_upper", "edge_lower"]].reset_index()
    bars_reset = bars.reset_index().rename(columns={"datetime": "ts"})

    merged = pd.merge_asof(
        bars_reset.sort_values("ts"), ref.sort_values("end_time"),
        left_on="ts", right_on="end_time", direction="backward"
    ).set_index("ts")

    events = []
    for ts, row in merged.iterrows():
        if row["phase"] != "lateralita" or pd.isna(row["edge_upper"]):
            continue

        # wick sopra il bordo superiore, chiusura rientrata dentro il box
        if row["high"] > row["edge_upper"] and row["close"] <= row["edge_upper"]:
            wick_size = row["high"] - row["edge_upper"]
            if wick_size < min_wick:
                continue
            confirmed = row["cvd_delta"] > 0  # comprato in aggressione sul breakout, ma respinto
            events.append({
                "datetime": ts, "side": "top", "edge": row["edge_upper"],
                "extreme": row["high"], "close": row["close"], "wick_size": wick_size,
                "cvd_delta": row["cvd_delta"], "confirmed_by_cvd": confirmed,
            })

        # wick sotto il bordo inferiore, chiusura rientrata dentro il box
        if row["low"] < row["edge_lower"] and row["close"] >= row["edge_lower"]:
            wick_size = row["edge_lower"] - row["low"]
            if wick_size < min_wick:
                continue
            confirmed = row["cvd_delta"] < 0  # venduto in aggressione sul breakdown, ma respinto
            events.append({
                "datetime": ts, "side": "bottom", "edge": row["edge_lower"],
                "extreme": row["low"], "close": row["close"], "wick_size": wick_size,
                "cvd_delta": row["cvd_delta"], "confirmed_by_cvd": confirmed,
            })

    return pd.DataFrame(events)


def summarize_transitions(result: pd.DataFrame):
    changes = result["phase"].ne(result["phase"].shift())
    transitions = result[changes]
    print("Transizioni di fase:")
    for ts, row in transitions.iterrows():
        print(f"  {ts}  ->  {row['phase'].upper()}  (close={row['close']:.2f}, z={row['z_velocity']:.2f})")


def summarize_absorptions(events: pd.DataFrame):
    if events.empty:
        print("\nNessun assorbimento ai bordi rilevato nel periodo.")
        return
    print(f"\nAssorbimenti ai bordi rilevati: {len(events)}")
    for _, e in events.iterrows():
        tag = "CONFERMATO" if e["confirmed_by_cvd"] else "non confermato"
        print(f"  {e['datetime']}  [{e['side'].upper()}]  bordo={e['edge']:.2f}  "
              f"estremo={e['extreme']:.2f}  chiusura={e['close']:.2f}  "
              f"wick={e['wick_size']:.2f}  cvd_delta={e['cvd_delta']:+.0f}  ({tag})")


def plot_result(result: pd.DataFrame, events: pd.DataFrame, out_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result.index, result["close"], color="black", linewidth=1)

    in_impulso = result["phase"] == "impulso"
    ax.fill_between(result.index, result["close"].min(), result["close"].max(),
                     where=in_impulso, color="orange", alpha=0.15, label="Impulso")
    ax.fill_between(result.index, result["close"].min(), result["close"].max(),
                     where=~in_impulso, color="steelblue", alpha=0.10, label="Lateralita'")

    ax.step(result.index, result["edge_upper"], where="post", color="green",
             linewidth=1, linestyle="--", label="Bordo box")
    ax.step(result.index, result["edge_lower"], where="post", color="green",
             linewidth=1, linestyle="--")

    if not events.empty:
        confirmed = events[events["confirmed_by_cvd"]]
        unconfirmed = events[~events["confirmed_by_cvd"]]
        ax.scatter(confirmed["datetime"], confirmed["extreme"], marker="^", color="red",
                   s=60, zorder=5, label="Assorbimento (CVD conferma)")
        ax.scatter(unconfirmed["datetime"], unconfirmed["extreme"], marker="x", color="gray",
                   s=40, zorder=5, label="Assorbimento (non confermato)")

    ax.set_title("Fasi di mercato e assorbimenti ai bordi (30m + 5m/CVD)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--start", required=True,
                         help="Data/ora da cui iniziare l'analisi, es. '2026-10-02 09:00:00'")
    parser.add_argument("--lookback", type=int, default=20,
                         help="Barre (30m) per calcolare la volatilita' di riferimento")
    parser.add_argument("--impulse-z", type=float, default=1.3,
                         help="Soglia z-score sopra cui una barra e' 'veloce'")
    parser.add_argument("--lateral-z", type=float, default=0.6,
                         help="Soglia z-score sotto cui una barra e' 'lenta'")
    parser.add_argument("--confirm-bars", type=int, default=2,
                         help="Barre consecutive richieste per confermare un cambio fase")
    parser.add_argument("--out", default=None, help="Path immagine output (es. fasi.png)")
    parser.add_argument("--events-out", default=None,
                         help="Path CSV dove salvare gli eventi di assorbimento (opzionale)")
    parser.add_argument("--min-wick", type=float, default=0.0,
                         help="Ampiezza minima della wick fuori bordo per contare come evento (filtra rumore)")
    parser.add_argument("--min-phase-bars", type=int, default=1,
                         help="Barre (30m) minime per considerare valida una fase; sotto, viene riassorbita nella fase precedente")
    args = parser.parse_args()

    df_5m = load_data(args.csv_path)
    ohlc_30m = resample_30m(df_5m)

    start_ts = pd.Timestamp(args.start)
    # calibra la volatilita'/box anche su dati precedenti allo start, poi taglia l'output
    result_full = classify_phases(
        ohlc_30m, args.lookback, args.impulse_z, args.lateral_z, args.confirm_bars
    )
    result_full = apply_min_duration(result_full, args.min_phase_bars)
    result_full = build_running_edges(result_full)

    result = result_full.loc[result_full.index >= start_ts]
    if result.empty:
        print("Nessun dato disponibile da/dopo la data di inizio indicata.")
        return

    print(f"Analisi da {start_ts} a {result.index[-1]} ({len(result)} barre da 30m)\n")
    summarize_transitions(result)

    refined = refine_signal_start(df_5m, result_full)
    refined = refined[refined["bar_30m_end"] >= start_ts] if not refined.empty else refined
    if not refined.empty:
        print("\nInizio segnale (raffinato sulle 5m):")
        for _, r in refined.iterrows():
            print(f"  barra 30m chiusa {r['bar_30m_end']}  ({r['direction']})")
            print(f"    assorbimento: {r['pivot_time']}  (close={r['pivot_close']:.2f})")
            if r["breakout_time"] is not None:
                print(f"    breakout confermato: {r['breakout_time']}  (close={r['breakout_close']:.2f})")
            else:
                print(f"    breakout confermato: n/d (nessun bordo di riferimento)")

    current = result.iloc[-1]
    print(f"\nFase attuale: {current['phase'].upper()}  "
          f"(ultima chiusura {current.name}: {current['close']:.2f}, z={current['z_velocity']:.2f})")

    events_full = detect_absorptions(df_5m, result_full, min_wick=args.min_wick)
    events = events_full[events_full["datetime"] >= start_ts] if not events_full.empty else events_full
    summarize_absorptions(events)

    if args.events_out and not events.empty:
        events.to_csv(args.events_out, index=False)
        print(f"\nEventi salvati in {args.events_out}")

    plot_result(result, events, out_path=args.out)


if __name__ == "__main__":
    main()
