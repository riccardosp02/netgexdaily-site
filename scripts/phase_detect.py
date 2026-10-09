"""
Identificazione fase di mercato (lateralita' / impulso) su candele 5m,
usando le chiusure, man mano che il tempo avanza. Fasi e bordo del box
sono calcolati entrambi sugli stessi 5m (nessun timeframe piu' alto).
Quando una fase e' lateralita', cerca assorbimenti/breakout ai bordi
del box, confermati dal CVD, e puo' confrontarli con quelli trovati
sulla Value Area (VAH/VAL) se fornita.

Uso:
    python3 scripts/phase_detect.py data/ohlc_cvd_2026-10.txt --start "2026-10-07 00:00:00" \
        --va-path data/value_area.csv

CSV atteso con colonne: date,time,open,high,low,close,cvd,vwap (5m)

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


def load_value_area(path: str) -> pd.DataFrame:
    """
    Carica un CSV di Value Area (date,time,price,vah,val,volume). vah/val
    sono gia' calcolati causalmente barra per barra (nessun dato futuro),
    quindi si possono usare direttamente come bordo "corto" e reattivo,
    alternativo al box 30m accumulato. Rimuove eventuali righe duplicate.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop_duplicates(subset="datetime").set_index("datetime").sort_index()
    return df[["price", "vah", "val", "volume"]]


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    OHLC(+cvd) ricampionato su un timeframe qualsiasi (es. '15min', '30min').
    Il bin [T, T+freq) viene etichettato con end_time = T+freq, cioe' il
    momento in cui la barra e' effettivamente chiusa e disponibile.
    """
    agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "cvd" in df.columns:
        agg_map["cvd"] = "last"
    agg = df.resample(freq, label="right", closed="left").agg(agg_map).dropna(subset=["close"])
    agg = agg.rename_axis("end_time")
    return agg


def resample_30m(df: pd.DataFrame) -> pd.DataFrame:
    """Alias storico: OHLC a 30 minuti (vedi resample_ohlc)."""
    return resample_ohlc(df, "30min")


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


def detect_va_events(df_5m: pd.DataFrame, va: pd.DataFrame,
                      result_30m: pd.DataFrame = None) -> pd.DataFrame:
    """
    Usa VAH/VAL (gia' causali, aggiornati ogni 5m) come bordo di
    riferimento al posto del box 30m accumulato.

    Riporta solo le TRANSIZIONI (prima barra fuori dopo un tratto dentro
    il VA), non ogni barra: altrimenti, essendo il VA molto stretto,
    segnala quasi in continuazione.

      - assorbimento: wick (high/low) fuori dal bordo, chiusura rientrata
        dentro
      - breakout: prima chiusura che esce dal bordo dopo essere stata
        dentro nella barra precedente

    Se result_30m e' passato, include solo eventi mentre la fase (30m,
    nota fino a quel momento) e' 'lateralita'. Il delta di CVD, se
    presente in df_5m, conferma la direzione.
    """
    merged = df_5m.join(va[["vah", "val"]], how="inner")
    if "cvd" in merged.columns:
        merged["cvd_delta"] = merged["cvd"].diff()
    else:
        merged["cvd_delta"] = np.nan

    if result_30m is not None:
        ref = result_30m[["phase"]].reset_index()
        ref = ref.rename(columns={ref.columns[0]: "ref_time"})
        merged = pd.merge_asof(
            merged.reset_index().rename(columns={"datetime": "ts"}).sort_values("ts"),
            ref.sort_values("ref_time"), left_on="ts", right_on="ref_time", direction="backward"
        ).set_index("ts")
        merged = merged[merged["phase"] == "lateralita"]

    was_above = merged["close"].shift() > merged["vah"].shift()
    was_below = merged["close"].shift() < merged["val"].shift()

    events = []
    for ts, row in merged.iterrows():
        if row["high"] > row["vah"] and row["close"] <= row["vah"]:
            events.append({"datetime": ts, "type": "assorbimento", "side": "top",
                            "edge": row["vah"], "extreme": row["high"], "close": row["close"],
                            "cvd_delta": row["cvd_delta"]})
        if row["low"] < row["val"] and row["close"] >= row["val"]:
            events.append({"datetime": ts, "type": "assorbimento", "side": "bottom",
                            "edge": row["val"], "extreme": row["low"], "close": row["close"],
                            "cvd_delta": row["cvd_delta"]})
        if row["close"] > row["vah"] and not was_above.loc[ts]:
            events.append({"datetime": ts, "type": "breakout", "side": "top",
                            "edge": row["vah"], "extreme": row["close"], "close": row["close"],
                            "cvd_delta": row["cvd_delta"]})
        elif row["close"] < row["val"] and not was_below.loc[ts]:
            events.append({"datetime": ts, "type": "breakout", "side": "bottom",
                            "edge": row["val"], "extreme": row["close"], "close": row["close"],
                            "cvd_delta": row["cvd_delta"]})

    return pd.DataFrame(events)


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


def find_box_to_box_moves(df_tf: pd.DataFrame, result: pd.DataFrame,
                           events: pd.DataFrame) -> pd.DataFrame:
    """
    Scansiona l'intero periodo e trova ogni passaggio
    lateralita' (box A) -> impulso (breakout) -> lateralita' (box B),
    cioe' il pattern "spostamento da una lateralita' all'altra".

    Per ciascuno misura:
      - bordi e durata del box di partenza e di quello di arrivo
      - timestamp/prezzo del PRIMO breakout che lascia il box di partenza
      - distanza percorsa dal breakout al bordo piu' vicino del box di
        arrivo (quanto "cammina" il prezzo tra una lateralita' e l'altra)
      - 'volume' (qui CVD, unico proxy disponibile nei dati OHLC): il
        |cvd_delta| medio durante l'impulso confrontato con quello
        durante il box di partenza -> volume_ratio > 1 vuol dire che
        l'impulso e' stato sostenuto da piu' attivita' che la
        lateralita' precedente, un indizio che il movimento era "vero"
        e non rumore.
    """
    phase = result["phase"]
    seg_id = phase.ne(phase.shift()).cumsum()
    segments = []
    for sid, idx in result.groupby(seg_id).groups.items():
        seg = result.loc[idx]
        segments.append({"phase": seg["phase"].iloc[0], "start": seg.index[0], "end": seg.index[-1]})

    moves = []
    for i in range(len(segments) - 2):
        box_a, impulso, box_b = segments[i], segments[i + 1], segments[i + 2]
        if box_a["phase"] != "lateralita" or impulso["phase"] != "impulso" or box_b["phase"] != "lateralita":
            continue

        seg_a = result.loc[box_a["start"]:box_a["end"]]
        seg_b = result.loc[box_b["start"]:box_b["end"]]
        a_upper, a_lower = seg_a["high"].max(), seg_a["low"].min()
        b_upper, b_lower = seg_b["high"].max(), seg_b["low"].min()

        bo = events[(events["type"] == "breakout") & (events["datetime"] >= impulso["start"]) &
                    (events["datetime"] <= impulso["end"])].sort_values("datetime")
        if bo.empty:
            continue
        first_bo = bo.iloc[0]

        if first_bo["side"] == "top":
            distance = b_lower - first_bo["close"]  # quanto manca al bordo piu' vicino del box B (dall'alto)
            direction = "up"
        else:
            distance = first_bo["close"] - b_upper
            direction = "down"

        impulso_bars = df_tf.loc[impulso["start"]:impulso["end"]]
        box_a_bars = df_tf.loc[box_a["start"]:box_a["end"]]
        if "cvd" in df_tf.columns:
            vol_impulso = impulso_bars["cvd"].diff().abs().mean()
            vol_box_a = box_a_bars["cvd"].diff().abs().mean()
            volume_ratio = vol_impulso / vol_box_a if vol_box_a else np.nan
        else:
            volume_ratio = np.nan

        moves.append({
            "box_a_start": box_a["start"], "box_a_end": box_a["end"],
            "box_a_upper": a_upper, "box_a_lower": a_lower,
            "breakout_time": first_bo["datetime"], "breakout_side": first_bo["side"],
            "breakout_close": first_bo["close"], "direction": direction,
            "box_b_start": box_b["start"], "box_b_end": box_b["end"],
            "box_b_upper": b_upper, "box_b_lower": b_lower,
            "distance": distance, "volume_ratio": volume_ratio,
        })

    return pd.DataFrame(moves)


def detect_absorptions_native(df_5m: pd.DataFrame, result_5m: pd.DataFrame,
                               min_wick: float = 0.0) -> pd.DataFrame:
    """
    Versione del box che usa le stesse barre 5m sia per classificare le
    fasi sia per cercare assorbimenti/breakout (nessun timeframe piu'
    alto di riferimento). Per restare causale, il bordo usato per
    valutare la barra t e' quello accumulato fino alla barra t-1
    (shiftato di una barra: una barra non puo' rompere un bordo che ha
    appena esteso lei stessa).
    """
    edge_upper_prior = result_5m["edge_upper"].shift()
    edge_lower_prior = result_5m["edge_lower"].shift()
    phase_prior = result_5m["phase"].shift()

    merged = df_5m.copy()
    merged["cvd_delta"] = merged["cvd"].diff() if "cvd" in merged.columns else np.nan
    merged["edge_upper"] = edge_upper_prior
    merged["edge_lower"] = edge_lower_prior
    merged["phase_prior"] = phase_prior

    events = []
    for ts, row in merged.iterrows():
        if row["phase_prior"] != "lateralita" or pd.isna(row["edge_upper"]):
            continue

        if row["high"] > row["edge_upper"] and row["close"] <= row["edge_upper"]:
            wick_size = row["high"] - row["edge_upper"]
            if wick_size >= min_wick:
                events.append({"datetime": ts, "type": "assorbimento", "side": "top",
                                "edge": row["edge_upper"], "extreme": row["high"],
                                "close": row["close"], "wick_size": wick_size,
                                "cvd_delta": row["cvd_delta"]})
        if row["low"] < row["edge_lower"] and row["close"] >= row["edge_lower"]:
            wick_size = row["edge_lower"] - row["low"]
            if wick_size >= min_wick:
                events.append({"datetime": ts, "type": "assorbimento", "side": "bottom",
                                "edge": row["edge_lower"], "extreme": row["low"],
                                "close": row["close"], "wick_size": wick_size,
                                "cvd_delta": row["cvd_delta"]})
        if row["close"] > row["edge_upper"]:
            events.append({"datetime": ts, "type": "breakout", "side": "top",
                            "edge": row["edge_upper"], "extreme": row["close"],
                            "close": row["close"], "wick_size": np.nan,
                            "cvd_delta": row["cvd_delta"]})
        elif row["close"] < row["edge_lower"]:
            events.append({"datetime": ts, "type": "breakout", "side": "bottom",
                            "edge": row["edge_lower"], "extreme": row["close"],
                            "close": row["close"], "wick_size": np.nan,
                            "cvd_delta": row["cvd_delta"]})

    return pd.DataFrame(events)


def add_confirmations(events: pd.DataFrame, va: pd.DataFrame = None,
                       vol_lookback: int = 6) -> pd.DataFrame:
    """
    Arricchisce gli eventi (assorbimento/breakout, qualunque sia la loro
    origine: box nativo 5m o Value Area) con due controlli aggiuntivi,
    che OGGI non influenzano la classificazione dell'evento (quella
    dipende solo dal prezzo: chiusura dentro/fuori dal bordo):

      - cvd_confirmed: il delta di CVD della barra spinge nella stessa
        direzione dell'evento (side='top' -> cvd_delta>0 comprato
        aggressivo; side='bottom' -> cvd_delta<0 venduto aggressivo).
        Per un breakout, "confermato" vuol dire che chi ha rotto il
        livello lo ha fatto spingendo davvero in quella direzione, non
        per inerzia/rumore. Per un assorbimento, vuol dire che c'era
        davvero un'aggressione (nella direzione del bordo) che e' stata
        respinta, non solo una wick senza volume dietro.

      - volume_trend: se e' disponibile una colonna 'volume' (tipicamente
        dal file Value Area), confronta il volume della barra con la
        media delle vol_lookback barre PRECEDENTI (causale, mai la barra
        stessa) e segna 'crescente' o 'calante'.
    """
    out = events.copy()
    if out.empty:
        out["cvd_confirmed"] = []
        out["volume_trend"] = []
        out["volume"] = []
        return out

    expected_sign = out["side"].map({"top": 1, "bottom": -1})
    out["cvd_confirmed"] = (np.sign(out["cvd_delta"]) == expected_sign)

    if va is not None and "volume" in va.columns:
        vol = va["volume"]
        vol_avg_prior = vol.rolling(vol_lookback).mean().shift()
        vol_at_ts = out["datetime"].map(vol)
        avg_at_ts = out["datetime"].map(vol_avg_prior)
        out["volume"] = vol_at_ts
        out["volume_trend"] = np.where(vol_at_ts > avg_at_ts, "crescente",
                                 np.where(vol_at_ts < avg_at_ts, "calante", "stabile"))
    else:
        out["volume"] = np.nan
        out["volume_trend"] = "n/d"

    return out


def evaluate_followthrough(events: pd.DataFrame, df_5m: pd.DataFrame,
                            horizons=(6, 12)) -> pd.DataFrame:
    """
    Per ogni evento di tipo 'breakout', guarda cosa fa il prezzo DOPO
    (solo barre successive all'evento, nessun look-ahead nella
    generazione del segnale: qui si valuta solo l'esito a posteriori).

    Per ogni orizzonte (in barre 5m, es. 6 = 30 min), calcola il
    movimento nella direzione del breakout:
      side='top'    -> close[t+h] - close[t]   (positivo = prosegue su)
      side='bottom' -> close[t] - close[t+h]   (positivo = prosegue giu')

    'prosegue' = movimento positivo (il breakout continua).
    """
    closes = df_5m["close"]
    rows = []
    for _, ev in events[events["type"] == "breakout"].iterrows():
        ts = ev["datetime"]
        if ts not in closes.index:
            continue
        pos = closes.index.get_loc(ts)
        row = {"datetime": ts, "side": ev["side"], "close": ev["close"],
               "cvd_confirmed": ev.get("cvd_confirmed"), "volume_trend": ev.get("volume_trend")}
        for h in horizons:
            if pos + h < len(closes):
                future_close = closes.iloc[pos + h]
                move = (future_close - ev["close"]) if ev["side"] == "top" else (ev["close"] - future_close)
                row[f"move_{h}b"] = move
                row[f"prosegue_{h}b"] = move > 0
            else:
                row[f"move_{h}b"] = np.nan
                row[f"prosegue_{h}b"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    for h in horizons:
        col = f"prosegue_{h}b"
        if col in out.columns:
            out[col] = out[col].astype("boolean")  # nullable bool: mean() ignora i NaN correttamente
    if "cvd_confirmed" in out.columns:
        out["cvd_confirmed"] = out["cvd_confirmed"].astype("boolean")
    return out


def analyze_absorption_sequences(events: pd.DataFrame, side: str = "top",
                                  threshold: int = 3) -> pd.DataFrame:
    """
    Scorre gli eventi in ordine cronologico contando gli assorbimenti
    consecutivi sullo stesso lato (es. 'top'). Il contatore si azzera
    quando arriva un breakout sul lato OPPOSTO (es. 'bottom': la rottura
    che l'assorbimento ripetuto in teoria dovrebbe anticipare/impedire).

    Per ogni volta che il contatore raggiunge 'threshold', registra:
      - quanti ULTERIORI assorbimenti sullo stesso lato arrivano prima
        della rottura (0 se la rottura arriva subito dopo la soglia)
      - come si risolve: 'opposite' (rottura dal lato opposto, l'esito
        "atteso"), 'same_side' (il bordo cede dallo STESSO lato degli
        assorbimenti: la sequenza "fallisce"), o 'open' (nessuna rottura
        nel periodo analizzato, sequenza ancora in corso)
    """
    if events.empty:
        return pd.DataFrame()

    opposite = "bottom" if side == "top" else "top"
    ev = events[events["type"].isin(["assorbimento", "breakout"])].sort_values("datetime")

    sequences = []
    run_count = 0
    run_start = None
    extra_after_threshold = 0
    triggered = False

    for _, row in ev.iterrows():
        if row["type"] == "assorbimento" and row["side"] == side:
            run_count += 1
            if run_count == 1:
                run_start = row["datetime"]
            if run_count == threshold:
                triggered = True
                extra_after_threshold = 0
            elif run_count > threshold and triggered:
                extra_after_threshold += 1

        elif row["type"] == "breakout" and row["side"] == opposite:
            if triggered:
                sequences.append({
                    "run_start": run_start, "threshold_reached_at_count": threshold,
                    "extra_absorptions_after_threshold": extra_after_threshold,
                    "total_absorptions_in_run": run_count,
                    "breakout_time": row["datetime"], "breakout_close": row["close"],
                    "outcome": "opposite",
                })
            run_count = 0
            run_start = None
            triggered = False
            extra_after_threshold = 0

        elif row["type"] == "breakout" and row["side"] == side:
            # breakout nella stessa direzione degli assorbimenti: la sequenza
            # "fallisce" (il bordo infine cede dallo stesso lato)
            if triggered:
                sequences.append({
                    "run_start": run_start, "threshold_reached_at_count": threshold,
                    "extra_absorptions_after_threshold": extra_after_threshold,
                    "total_absorptions_in_run": run_count,
                    "breakout_time": row["datetime"], "breakout_close": row["close"],
                    "outcome": "same_side",
                })
            run_count = 0
            run_start = None
            triggered = False
            extra_after_threshold = 0

    if triggered:
        sequences.append({
            "run_start": run_start, "threshold_reached_at_count": threshold,
            "extra_absorptions_after_threshold": extra_after_threshold,
            "total_absorptions_in_run": run_count,
            "breakout_time": None, "breakout_close": None,
            "outcome": "open",
        })

    return pd.DataFrame(sequences)


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
    """Riepiloga eventi nel formato di detect_absorptions_native/detect_va_events
    (colonne: datetime, type, side, edge, extreme, close, wick_size, cvd_delta)."""
    if events.empty:
        print("\nNessun evento (assorbimento/breakout) rilevato nel periodo.")
        return
    print(f"\nEventi ai bordi rilevati: {len(events)}")
    for _, e in events.iterrows():
        wick = f"wick={e['wick_size']:.2f}  " if pd.notna(e.get("wick_size")) else ""
        cvd = f"cvd_delta={e['cvd_delta']:+.0f}" if pd.notna(e.get("cvd_delta")) else "cvd_delta=n/d"
        print(f"  {e['datetime']}  [{e['type'].upper()} {e['side'].upper()}]  bordo={e['edge']:.2f}  "
              f"estremo={e['extreme']:.2f}  chiusura={e['close']:.2f}  {wick}{cvd}")


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
        absorb = events[events["type"] == "assorbimento"]
        breakout = events[events["type"] == "breakout"]
        ax.scatter(absorb["datetime"], absorb["extreme"], marker="v", color="red",
                   s=60, zorder=5, label="Assorbimento")
        ax.scatter(breakout["datetime"], breakout["close"], marker="x", color="blue",
                   s=40, zorder=5, label="Breakout")

    ax.set_title("Fasi di mercato e assorbimenti/breakout ai bordi (box nativo 5m)")
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
                         help="Data/ora da cui iniziare l'analisi, es. '2026-10-07 00:00:00'")
    parser.add_argument("--timeframe", default="15min",
                         help="Timeframe nativo per fasi e box (es. '5min', '15min', '30min')")
    parser.add_argument("--lookback", type=int, default=15,
                         help="Barre (al timeframe scelto) per calcolare la volatilita' di riferimento")
    parser.add_argument("--impulse-z", type=float, default=1.2,
                         help="Soglia z-score sopra cui una barra e' 'veloce'")
    parser.add_argument("--lateral-z", type=float, default=0.5,
                         help="Soglia z-score sotto cui una barra e' 'lenta'")
    parser.add_argument("--confirm-bars", type=int, default=2,
                         help="Barre consecutive richieste per confermare un cambio fase")
    parser.add_argument("--min-phase-bars", type=int, default=2,
                         help="Barre minime per considerare valida una fase; sotto, viene riassorbita nella fase precedente")
    parser.add_argument("--min-wick", type=float, default=0.0,
                         help="Ampiezza minima della wick fuori bordo per contare come evento (filtra rumore)")
    parser.add_argument("--va-path", default=None,
                         help="CSV Value Area (date,time,price,vah,val,volume), opzionale, per confronto")
    parser.add_argument("--out", default=None, help="Path immagine output (es. fasi.png)")
    parser.add_argument("--events-out", default=None,
                         help="Path CSV dove salvare gli eventi di assorbimento (opzionale)")
    parser.add_argument("--sequence-side", default="top", choices=["top", "bottom"],
                         help="Lato per l'analisi delle sequenze di assorbimento")
    parser.add_argument("--sequence-threshold", type=int, default=3,
                         help="Quanti assorbimenti consecutivi sullo stesso lato prima di iniziare a contare")
    args = parser.parse_args()

    df_5m = load_data(args.csv_path)
    df_tf = resample_ohlc(df_5m, args.timeframe) if args.timeframe != "5min" else df_5m
    start_ts = pd.Timestamp(args.start)

    # tutto allo stesso timeframe nativo: fasi e box usano le stesse barre
    result_full = classify_phases(
        df_tf, args.lookback, args.impulse_z, args.lateral_z, args.confirm_bars
    )
    result_full = apply_min_duration(result_full, args.min_phase_bars)
    result_full = build_running_edges(result_full)

    result = result_full.loc[result_full.index >= start_ts]
    if result.empty:
        print("Nessun dato disponibile da/dopo la data di inizio indicata.")
        return

    print(f"Analisi da {start_ts} a {result.index[-1]} ({len(result)} barre da {args.timeframe})\n")
    summarize_transitions(result)

    current = result.iloc[-1]
    print(f"\nFase attuale: {current['phase'].upper()}  "
          f"(ultima chiusura {current.name}: {current['close']:.2f}, z={current['z_velocity']:.2f})")

    events_full = detect_absorptions_native(df_tf, result_full, min_wick=args.min_wick)
    events = events_full[events_full["datetime"] >= start_ts] if not events_full.empty else events_full
    summarize_absorptions(events)

    if args.events_out and not events.empty:
        events.to_csv(args.events_out, index=False)
        print(f"\nEventi salvati in {args.events_out}")

    seq = analyze_absorption_sequences(events, side=args.sequence_side, threshold=args.sequence_threshold)
    print(f"\nSequenze di assorbimento ({args.sequence_threshold}+ consecutivi lato {args.sequence_side.upper()}):")
    print(seq.to_string(index=False) if not seq.empty else "  nessuna")

    if args.va_path:
        va = load_value_area(args.va_path)
        va_events_full = detect_va_events(df_5m, va, result_full)
        va_events = va_events_full[va_events_full["datetime"] >= start_ts] if not va_events_full.empty else va_events_full
        print(f"\nEventi Value Area: {len(va_events)}")
        seq_va = analyze_absorption_sequences(va_events, side=args.sequence_side, threshold=args.sequence_threshold)
        print(f"Sequenze di assorbimento VA ({args.sequence_threshold}+ consecutivi lato {args.sequence_side.upper()}):")
        print(seq_va.to_string(index=False) if not seq_va.empty else "  nessuna")

    plot_result(result, events, out_path=args.out)


if __name__ == "__main__":
    main()
