from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data_sources import (
    MatchEvent,
    events_to_dataframe,
    get_football_data_matches,
    get_odds_events,
    get_odds_sports,
    manual_events_from_dataframe,
)
from core.predictor import estimate_xg_from_odds, simulate_match, result_summary_df
from core.ai_analyzer import gemini_analysis
from core.utils import decimal_odds_product, fmt_money, fmt_pct, safe_float, utc_iso_range

load_dotenv()

st.set_page_config(
    page_title="Bot Predictor Automático",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.stMetric {border: 1px solid rgba(128,128,128,.18); border-radius: 14px; padding: 12px;}
.small-note {font-size: .88rem; opacity: .78;}
.pick-box {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 14px; margin: 6px 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚽ Bot Predictor Automático de Fútbol")
st.caption("Busca partidos, trae cuotas, simula miles de escenarios y genera pronósticos. No garantiza resultados; solo estima probabilidades.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔑 APIs")
    odds_key = st.text_input("The Odds API Key", value=os.getenv("ODDS_API_KEY", ""), type="password")
    football_key = st.text_input("Football-Data API Key", value=os.getenv("FOOTBALL_DATA_API_KEY", ""), type="password")
    gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    gemini_model = st.text_input("Modelo Gemini", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    use_google_search = st.checkbox("Gemini con búsqueda de Google", value=True)

    st.divider()
    st.header("💵 Banca")
    bank = st.number_input("Banca disponible", min_value=1.0, value=3.0, step=0.5)
    simulations = st.slider("Simulaciones Monte Carlo", 5000, 200000, 50000, step=5000)

    st.divider()
    st.header("🌍 Buscar partidos")
    source = st.selectbox("Fuente", ["The Odds API", "Football-Data", "Manual CSV"])

if "events" not in st.session_state:
    st.session_state.events = []
if "sports" not in st.session_state:
    st.session_state.sports = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------- Auto search tabs ----------------
tab_search, tab_analyze, tab_ticket, tab_help = st.tabs(["🔎 Buscar partidos", "📊 Analizar partido", "🎟️ Generar ticket", "🛠️ GitHub y ayuda"])

with tab_search:
    st.subheader("Buscar partidos automáticamente")

    if source == "The Odds API":
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            regions = st.selectbox("Región de cuotas", ["eu", "uk", "us", "us2", "au"], index=0)
        with colB:
            markets = st.multiselect("Mercados", ["h2h", "totals"], default=["h2h"])
        with colC:
            preferred_bookmaker = st.text_input("Bookmaker preferido opcional", placeholder="Ej: Bet365, Pinnacle, Betfair")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            days_ahead = st.number_input("Días hacia adelante", min_value=1, max_value=30, value=3)
        with c2:
            only_soccer = st.checkbox("Solo fútbol", value=True)
        with c3:
            st.write("")
            st.write("")
            load_sports = st.button("1) Cargar ligas/deportes", use_container_width=True)

        if load_sports:
            try:
                st.session_state.sports = get_odds_sports(odds_key, only_soccer=only_soccer)
                st.success(f"Se cargaron {len(st.session_state.sports)} deportes/ligas.")
            except Exception as exc:
                st.error(str(exc))

        sports = st.session_state.sports
        if sports:
            labels = [f"{s.get('title')} — {s.get('key')}" for s in sports]
            selected_label = st.selectbox("Elige liga/deporte", labels)
            selected_sport = sports[labels.index(selected_label)]["key"]
        else:
            selected_sport = st.text_input("Sport key manual", value="soccer_fifa_world_cup")

        if st.button("2) Buscar partidos y cuotas", type="primary", use_container_width=True):
            try:
                start_iso, end_iso = utc_iso_range(int(days_ahead))
                events = get_odds_events(
                    api_key=odds_key,
                    sport_key=selected_sport,
                    regions=regions,
                    markets=",".join(markets) if markets else "h2h",
                    commence_from=start_iso,
                    commence_to=end_iso,
                    preferred_bookmaker=preferred_bookmaker or None,
                )
                st.session_state.events = events
                st.success(f"Encontré {len(events)} partidos con datos de cuotas.")
            except Exception as exc:
                st.error(str(exc))

    elif source == "Football-Data":
        col1, col2, col3 = st.columns(3)
        today = datetime.now(timezone.utc).date()
        with col1:
            date_from = st.date_input("Desde", value=today)
        with col2:
            date_to = st.date_input("Hasta", value=today + timedelta(days=3))
        with col3:
            competitions = st.text_input("Competiciones opcional", placeholder="PL,CL,SA,BL1,PD,WC")
        if st.button("Buscar partidos Football-Data", type="primary", use_container_width=True):
            try:
                events = get_football_data_matches(
                    football_key,
                    date_from=str(date_from),
                    date_to=str(date_to),
                    competitions=competitions or None,
                )
                st.session_state.events = events
                st.success(f"Encontré {len(events)} partidos.")
            except Exception as exc:
                st.error(str(exc))

    else:
        st.info("Sube un CSV con columnas: home_team, away_team, home_odds, draw_odds, away_odds")
        uploaded = st.file_uploader("CSV manual", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv("data/manual_matches.csv")
            st.caption("Usando ejemplo incluido.")
        st.dataframe(df, use_container_width=True)
        if st.button("Cargar partidos manuales", type="primary", use_container_width=True):
            st.session_state.events = manual_events_from_dataframe(df)
            st.success(f"Cargados {len(st.session_state.events)} partidos manuales.")

    st.divider()
    st.subheader("Partidos encontrados")
    events = st.session_state.events
    if events:
        df_events = events_to_dataframe(events)
        st.dataframe(df_events, use_container_width=True, hide_index=True)
        csv_bytes = df_events.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar lista CSV", csv_bytes, "partidos_encontrados.csv", "text/csv")
    else:
        st.warning("Todavía no hay partidos cargados. Busca con una API o carga CSV manual.")

with tab_analyze:
    st.subheader("Analizar el partido que tú elijas")
    events: List[MatchEvent] = st.session_state.events

    if not events:
        st.warning("Primero busca o carga partidos en la pestaña 🔎 Buscar partidos.")
    else:
        labels = [f"{i}. {e.home_team} vs {e.away_team} | {e.competition} | {e.commence_time}" for i, e in enumerate(events)]
        chosen = st.selectbox("Elige partido para analizar", labels)
        event = events[labels.index(chosen)]

        st.markdown(f"### {event.home_team} vs {event.away_team}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cuota local", event.home_odds or "N/D")
        c2.metric("Cuota empate", event.draw_odds or "N/D")
        c3.metric("Cuota visitante", event.away_odds or "N/D")
        c4.metric("Fuente", event.source)

        with st.expander("Ajustes avanzados de estimación", expanded=False):
            estimated_home_xg, estimated_away_xg, market_probs = estimate_xg_from_odds(
                event.home_odds,
                event.draw_odds,
                event.away_odds,
                event.over25_odds,
                event.under25_odds,
            )
            colx1, colx2 = st.columns(2)
            with colx1:
                home_xg = st.number_input(f"xG estimado {event.home_team}", min_value=0.1, max_value=5.0, value=float(estimated_home_xg), step=0.05)
            with colx2:
                away_xg = st.number_input(f"xG estimado {event.away_team}", min_value=0.1, max_value=5.0, value=float(estimated_away_xg), step=0.05)
            st.caption("El xG automático sale de las cuotas 1X2 y, si existe, del mercado Over/Under 2.5. Puedes corregirlo manualmente.")

        if st.button("🚀 Analizar y simular este partido", type="primary", use_container_width=True):
            odds_dict = {
                "home": event.home_odds,
                "draw": event.draw_odds,
                "away": event.away_odds,
                "over25": event.over25_odds,
            }
            result = simulate_match(
                home_team=event.home_team,
                away_team=event.away_team,
                home_xg=float(home_xg),
                away_xg=float(away_xg),
                simulations=int(simulations),
                bank=float(bank),
                odds=odds_dict,
            )
            st.session_state.last_result = {"event": event, "result": result}

        last = st.session_state.last_result
        if last and last["event"].event_id == event.event_id:
            result = last["result"]
            st.divider()
            st.subheader("Probabilidades principales")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Gana {event.home_team}", fmt_pct(result.p_home))
            m2.metric("Empate", fmt_pct(result.p_draw))
            m3.metric(f"Gana {event.away_team}", fmt_pct(result.p_away))
            m4.metric("Modelo", f"{result.simulations:,} sims")

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Más de 1.5", fmt_pct(result.p_over15))
            g2.metric("Más de 2.5", fmt_pct(result.p_over25))
            g3.metric("Menos de 3.5", fmt_pct(result.p_under35))
            g4.metric("Ambos marcan", fmt_pct(result.p_btts))

            st.subheader("Ranking de mercados")
            market_df = pd.DataFrame(result.market_table)
            st.dataframe(market_df, use_container_width=True, hide_index=True)

            st.subheader("Marcadores más probables")
            score_df = pd.DataFrame(result.top_scores, columns=["Marcador", "Probabilidad"])
            score_df["Probabilidad"] = (score_df["Probabilidad"] * 100).round(2).astype(str) + "%"
            st.dataframe(score_df, use_container_width=True, hide_index=True)

            st.subheader("Decisión del bot")
            colp1, colp2 = st.columns(2)
            with colp1:
                st.markdown("**Mejor pick por equilibrio valor/probabilidad**")
                st.json(result.best_pick)
            with colp2:
                st.markdown("**Pick más útil para ticket combinado**")
                st.json(result.ticket_pick)

            st.subheader("Análisis por áreas")
            area1, area2, area3 = st.columns(3)
            with area1:
                st.markdown("**1. Cuotas y valor**")
                st.write("El bot compara la probabilidad simulada con la probabilidad implícita de la cuota. Si la diferencia es positiva, puede existir valor.")
            with area2:
                st.markdown("**2. Riesgo**")
                st.write("Riesgo bajo si la probabilidad supera 72%, medio si supera 56%, alto por debajo de ese rango.")
            with area3:
                st.markdown("**3. Banca**")
                st.write("El stake sugerido evita meter toda la banca salvo que tú lo decidas manualmente.")

            if st.button("🧠 Generar análisis con Gemini", use_container_width=True):
                context = {
                    "partido": f"{event.home_team} vs {event.away_team}",
                    "competicion": event.competition,
                    "fecha": event.commence_time,
                    "cuotas": {"local": event.home_odds, "empate": event.draw_odds, "visitante": event.away_odds},
                    "xg": {event.home_team: result.home_xg, event.away_team: result.away_xg},
                    "probabilidades": {
                        "local": result.p_home,
                        "empate": result.p_draw,
                        "visitante": result.p_away,
                        "1X": result.p_1x,
                        "X2": result.p_x2,
                        "over15": result.p_over15,
                        "over25": result.p_over25,
                        "under35": result.p_under35,
                        "btts": result.p_btts,
                    },
                    "mejor_pick": result.best_pick,
                    "pick_ticket": result.ticket_pick,
                }
                text = gemini_analysis(gemini_key, gemini_model, context, use_google_search=use_google_search)
                st.markdown(text)
                st.download_button("Descargar análisis IA", text.encode("utf-8"), "analisis_gemini.txt", "text/plain")

with tab_ticket:
    st.subheader("Generador automático de ticket")
    events: List[MatchEvent] = st.session_state.events
    if not events:
        st.warning("Primero busca partidos.")
    else:
        max_legs = st.slider("Máximo de selecciones en el ticket", 1, 5, 3)
        min_prob = st.slider("Probabilidad mínima por selección", 50, 90, 62)
        use_safe_mode = st.checkbox("Modo conservador", value=True)

        if st.button("🎟️ Crear ticket automático", type="primary", use_container_width=True):
            picks = []
            for e in events:
                hxg, axg, _ = estimate_xg_from_odds(e.home_odds, e.draw_odds, e.away_odds, e.over25_odds, e.under25_odds)
                res = simulate_match(e.home_team, e.away_team, hxg, axg, simulations=min(int(simulations), 60000), bank=float(bank), odds={"home": e.home_odds, "draw": e.draw_odds, "away": e.away_odds, "over25": e.over25_odds})
                candidates = []
                for row in res.market_table:
                    prob = float(row["Prob. bot"])
                    if prob >= min_prob:
                        # En modo conservador preferimos mercados simples y doble oportunidad.
                        name = str(row["Mercado"])
                        safe_bonus = 0
                        if "empate" in name.lower() or "más de 1.5" in name.lower() or "menos de 3.5" in name.lower():
                            safe_bonus = 5
                        candidates.append((prob + safe_bonus, row, e))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    picks.append(candidates[0])
            picks.sort(key=lambda x: x[0], reverse=True)
            st.session_state.ticket_picks = picks[:max_legs]

        ticket = st.session_state.get("ticket_picks", [])
        if ticket:
            st.markdown("### Ticket sugerido")
            rows = []
            odds_list = []
            for score, row, e in ticket:
                odds_val = row.get("Cuota")
                if odds_val:
                    odds_list.append(float(odds_val))
                rows.append({
                    "Partido": f"{e.home_team} vs {e.away_team}",
                    "Selección": row["Mercado"],
                    "Probabilidad": f"{row['Prob. bot']}%",
                    "Riesgo": row["Riesgo"],
                    "Cuota": odds_val,
                    "Stake sugerido": row["Stake sugerido"],
                })
            df_ticket = pd.DataFrame(rows)
            st.dataframe(df_ticket, use_container_width=True, hide_index=True)
            total_odds = decimal_odds_product(odds_list) if odds_list else None
            if total_odds:
                ret = float(bank) * total_odds
                gain = ret - float(bank)
                c1, c2, c3 = st.columns(3)
                c1.metric("Cuota total aprox.", f"{total_odds:.2f}")
                c2.metric("Retorno con banca", fmt_money(ret))
                c3.metric("Ganancia limpia", fmt_money(gain))
            msg = "Bro, ponme este ticket:\n\n" + "\n".join([f"{r['Partido']}: {r['Selección']}" for r in rows]) + f"\n\nMonto: ${bank:.2f}"
            if total_odds:
                msg += f"\nCuota aprox.: {total_odds:.2f}\nRetorno aprox.: ${ret:.2f}"
            st.text_area("Mensaje listo para enviar", msg, height=180)

with tab_help:
    st.subheader("Subir a GitHub")
    st.write("Este proyecto trae un archivo `github_crear_y_subir.bat`. Con GitHub CLI instalado y logueado, crea el repo remoto y sube el proyecto automáticamente.")
    st.code("github_crear_y_subir.bat", language="bat")
    st.write("También puedes subirlo manualmente con `github_subir_manual.bat` pegando la URL del repo remoto.")

    st.subheader("Notas importantes")
    st.markdown(
        """
- The Odds API es la fuente principal para buscar partidos y cuotas.
- Football-Data sirve para calendario y partidos por competición, pero algunas ligas requieren plan de pago.
- Gemini puede generar análisis con búsqueda de Google si tu API/cuenta/modelo lo permite.
- El bot estima probabilidades; no existe apuesta segura.
"""
    )
