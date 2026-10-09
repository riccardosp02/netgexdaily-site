"""
Identificazione fase di mercato (lateralita' / impulso) su candele 30m,
usando le chiusure, man mano che il tempo avanza.

Uso:
    python3 scripts/phase_detect.py data/ohlc_cvd_2026-10.txt --start "2026-10-02 09:00:00"

CSV atteso con colonne: date,time,open,high,low,close,cvd,vwap
(timeframe nativo 5m; viene ricampionato a 30m usando le sole chiusure)

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


def resample_30m_closes(df: pd.DataFrame) -> pd.Series:
    """Chiusura ogni 30 minuti (ultima chiusura disponibile nel bucket)."""
    closes = df["close"].resample("30min").last().dropna()
    return closes


def classify_phases(closes: pd.Series, lookback: int, impulse_z: float,
                     lateral_z: float, confirm_bars: int) -> pd.DataFrame:
    """
    Per ogni barra calcola la "velocita'" del prezzo (variazione della chiusura
    rispetto alla barra precedente) normalizzata sulla volatilita' recente
    (rolling std dei ritorni, lookback barre).

    Stato con isteresi:
      - entra in IMPULSO se la velocita' (z-score) resta sopra impulse_z per
        confirm_bars barre consecutive
      - torna in LATERALITA' se resta sotto lateral_z per confirm_bars barre
        consecutive
      - altrimenti mantiene la fase corrente (evita sfarfallio)
    """
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

    out = pd.DataFrame({"close": closes, "z_velocity": z, "phase": phases})
    return out


def summarize_transitions(result: pd.DataFrame):
    changes = result["phase"].ne(result["phase"].shift())
    transitions = result[changes]
    print("Transizioni di fase:")
    for ts, row in transitions.iterrows():
        print(f"  {ts}  ->  {row['phase'].upper()}  (close={row['close']:.2f}, z={row['z_velocity']:.2f})")


def plot_result(result: pd.DataFrame, out_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result.index, result["close"], color="black", linewidth=1)

    in_impulso = result["phase"] == "impulso"
    ax.fill_between(result.index, result["close"].min(), result["close"].max(),
                     where=in_impulso, color="orange", alpha=0.15, label="Impulso")
    ax.fill_between(result.index, result["close"].min(), result["close"].max(),
                     where=~in_impulso, color="steelblue", alpha=0.10, label="Lateralita'")

    ax.set_title("Fasi di mercato: lateralita' vs impulso (chiusure 30m)")
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
    args = parser.parse_args()

    df = load_data(args.csv_path)
    closes_full = resample_30m_closes(df)

    start_ts = pd.Timestamp(args.start)
    # calibra la volatilita' anche su dati precedenti allo start, poi taglia l'output
    result_full = classify_phases(
        closes_full, args.lookback, args.impulse_z, args.lateral_z, args.confirm_bars
    )
    result = result_full.loc[result_full.index >= start_ts]

    if result.empty:
        print("Nessun dato disponibile da/dopo la data di inizio indicata.")
        return

    print(f"Analisi da {start_ts} a {result.index[-1]} ({len(result)} barre da 30m)\n")
    summarize_transitions(result)

    current = result.iloc[-1]
    print(f"\nFase attuale: {current['phase'].upper()}  "
          f"(ultima chiusura {current.name}: {current['close']:.2f}, z={current['z_velocity']:.2f})")

    plot_result(result, out_path=args.out)


if __name__ == "__main__":
    main()
