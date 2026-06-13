from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd

ODDS_BASE = "https://api.the-odds-api.com/v4"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"


@dataclass
class MatchEvent:
    source: str
    event_id: str
    sport_key: str
    competition: str
    home_team: str
    away_team: str
    commence_time: str
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    over25_odds: Optional[float] = None
    under25_odds: Optional[float] = None
    bookmaker: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data


def _get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 25) -> Any:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code == 401:
        raise RuntimeError("API Key inválida o no autorizada.")
    if resp.status_code == 429:
        raise RuntimeError("Límite de peticiones alcanzado. Espera o usa otra clave/API.")
    if resp.status_code >= 400:
        raise RuntimeError(f"Error HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def get_odds_sports(api_key: str, only_soccer: bool = True) -> List[Dict[str, Any]]:
    if not api_key:
        raise ValueError("Falta The Odds API Key.")
    data = _get_json(f"{ODDS_BASE}/sports", params={"apiKey": api_key})
    if only_soccer:
        data = [s for s in data if str(s.get("group", "")).lower() == "soccer" or str(s.get("key", "")).startswith("soccer")]
    return sorted(data, key=lambda x: (x.get("group", ""), x.get("title", "")))


def _extract_best_h2h_and_totals(event: Dict[str, Any], preferred_bookmaker: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    best_home = best_draw = best_away = None
    best_over25 = best_under25 = None
    book_used = None

    bookmakers = event.get("bookmakers", []) or []
    if preferred_bookmaker:
        preferred = [b for b in bookmakers if preferred_bookmaker.lower() in (b.get("title", "") + " " + b.get("key", "")).lower()]
        bookmakers = preferred + [b for b in bookmakers if b not in preferred]

    for book in bookmakers:
        title = book.get("title") or book.get("key")
        for market in book.get("markets", []) or []:
            key = market.get("key")
            outcomes = market.get("outcomes", []) or []
            if key == "h2h":
                for out in outcomes:
                    name = str(out.get("name", ""))
                    price = _safe_float(out.get("price"))
                    if price is None:
                        continue
                    lname = name.lower()
                    if name == home or home.lower() in lname or lname in home.lower():
                        best_home = max(best_home or 0, price)
                        book_used = book_used or title
                    elif name == away or away.lower() in lname or lname in away.lower():
                        best_away = max(best_away or 0, price)
                        book_used = book_used or title
                    elif lname in {"draw", "empate"}:
                        best_draw = max(best_draw or 0, price)
                        book_used = book_used or title
            elif key == "totals":
                for out in outcomes:
                    point = _safe_float(out.get("point"))
                    price = _safe_float(out.get("price"))
                    name = str(out.get("name", "")).lower()
                    if price is None or point is None:
                        continue
                    # Usamos 2.5 porque es el mercado más común para goles.
                    if abs(point - 2.5) < 0.01:
                        if "over" in name or "más" in name or "mas" in name:
                            best_over25 = max(best_over25 or 0, price)
                        elif "under" in name or "menos" in name:
                            best_under25 = max(best_under25 or 0, price)

    return best_home, best_draw, best_away, best_over25, best_under25, book_used


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def get_odds_events(
    api_key: str,
    sport_key: str,
    regions: str = "eu",
    markets: str = "h2h",
    odds_format: str = "decimal",
    commence_from: Optional[str] = None,
    commence_to: Optional[str] = None,
    preferred_bookmaker: Optional[str] = None,
) -> List[MatchEvent]:
    if not api_key:
        raise ValueError("Falta The Odds API Key.")
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if commence_from:
        params["commenceTimeFrom"] = commence_from
    if commence_to:
        params["commenceTimeTo"] = commence_to

    data = _get_json(f"{ODDS_BASE}/sports/{sport_key}/odds", params=params)
    events: List[MatchEvent] = []
    for ev in data:
        h, d, a, o25, u25, book = _extract_best_h2h_and_totals(ev, preferred_bookmaker)
        events.append(
            MatchEvent(
                source="The Odds API",
                event_id=str(ev.get("id", "")),
                sport_key=str(ev.get("sport_key", sport_key)),
                competition=str(ev.get("sport_title", sport_key)),
                home_team=str(ev.get("home_team", "Local")),
                away_team=str(ev.get("away_team", "Visitante")),
                commence_time=str(ev.get("commence_time", "")),
                home_odds=h,
                draw_odds=d,
                away_odds=a,
                over25_odds=o25,
                under25_odds=u25,
                bookmaker=book,
                raw=ev,
            )
        )
    return events


def get_football_data_matches(api_key: str, date_from: str, date_to: str, competitions: Optional[str] = None) -> List[MatchEvent]:
    if not api_key:
        raise ValueError("Falta Football-Data API Key.")
    headers = {"X-Auth-Token": api_key}
    params = {"dateFrom": date_from, "dateTo": date_to}
    if competitions:
        params["competitions"] = competitions
    data = _get_json(f"{FOOTBALL_DATA_BASE}/matches", params=params, headers=headers)
    events = []
    for m in data.get("matches", []) or []:
        home = (m.get("homeTeam") or {}).get("name") or "Local"
        away = (m.get("awayTeam") or {}).get("name") or "Visitante"
        comp = (m.get("competition") or {}).get("name") or "Football-Data"
        events.append(MatchEvent(
            source="Football-Data",
            event_id=str(m.get("id", "")),
            sport_key=str((m.get("competition") or {}).get("code", "")),
            competition=comp,
            home_team=home,
            away_team=away,
            commence_time=str(m.get("utcDate", "")),
            raw=m,
        ))
    return events


def manual_events_from_dataframe(df: pd.DataFrame) -> List[MatchEvent]:
    events: List[MatchEvent] = []
    for i, row in df.iterrows():
        events.append(MatchEvent(
            source="Manual CSV",
            event_id=f"manual-{i}",
            sport_key="manual",
            competition="Manual",
            home_team=str(row.get("home_team", "Local")),
            away_team=str(row.get("away_team", "Visitante")),
            commence_time="",
            home_odds=_safe_float(row.get("home_odds")),
            draw_odds=_safe_float(row.get("draw_odds")),
            away_odds=_safe_float(row.get("away_odds")),
        ))
    return events


def events_to_dataframe(events: List[MatchEvent]) -> pd.DataFrame:
    rows = []
    for idx, e in enumerate(events):
        rows.append({
            "idx": idx,
            "fuente": e.source,
            "competición": e.competition,
            "fecha_utc": e.commence_time,
            "partido": f"{e.home_team} vs {e.away_team}",
            "local": e.home_team,
            "visitante": e.away_team,
            "cuota_local": e.home_odds,
            "cuota_empate": e.draw_odds,
            "cuota_visitante": e.away_odds,
            "over25": e.over25_odds,
            "under25": e.under25_odds,
            "bookmaker": e.bookmaker,
        })
    return pd.DataFrame(rows)
