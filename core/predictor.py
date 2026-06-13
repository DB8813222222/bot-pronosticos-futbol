from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import Counter
import math
import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    simulations: int
    p_home: float
    p_draw: float
    p_away: float
    p_1x: float
    p_x2: float
    p_12: float
    p_over15: float
    p_over25: float
    p_under35: float
    p_btts: float
    top_scores: List[Tuple[str, float]]
    market_table: List[Dict[str, object]]
    best_pick: Dict[str, object]
    ticket_pick: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def implied_probability(odds: Optional[float]) -> Optional[float]:
    if not odds or odds <= 1:
        return None
    return 1.0 / odds


def normalize_1x2(home_odds: Optional[float], draw_odds: Optional[float], away_odds: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    probs = [implied_probability(home_odds), implied_probability(draw_odds), implied_probability(away_odds)]
    if any(p is None for p in probs):
        return None, None, None
    total = sum(p for p in probs if p is not None)
    if total <= 0:
        return None, None, None
    return probs[0] / total, probs[1] / total, probs[2] / total


def poisson_cdf(k: int, lam: float) -> float:
    # P(X <= k)
    return sum(math.exp(-lam) * (lam ** i) / math.factorial(i) for i in range(k + 1))


def estimate_total_goals_from_over25(p_over25: float) -> float:
    # P(total > 2.5) = 1 - P(total <= 2). Buscamos lambda total aproximado.
    p = min(max(p_over25, 0.05), 0.95)
    lo, hi = 0.5, 5.5
    for _ in range(60):
        mid = (lo + hi) / 2
        prob = 1 - poisson_cdf(2, mid)
        if prob < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def estimate_xg_from_odds(
    home_odds: Optional[float],
    draw_odds: Optional[float],
    away_odds: Optional[float],
    over25_odds: Optional[float] = None,
    under25_odds: Optional[float] = None,
    base_total: float = 2.55,
) -> Tuple[float, float, Dict[str, float]]:
    ph, pd, pa = normalize_1x2(home_odds, draw_odds, away_odds)
    if ph is None:
        ph, pd, pa = 0.45, 0.27, 0.28

    total_goals = base_total
    p_over = implied_probability(over25_odds)
    p_under = implied_probability(under25_odds)
    if p_over and p_under:
        s = p_over + p_under
        if s > 0:
            p_over_norm = p_over / s
            total_goals = estimate_total_goals_from_over25(p_over_norm)

    strength = ph - pa
    # Ajuste por localía más ventaja de mercado.
    home_xg = (total_goals / 2) + (0.95 * strength) + 0.12
    away_xg = total_goals - home_xg
    home_xg = min(max(home_xg, 0.25), 3.60)
    away_xg = min(max(away_xg, 0.20), 3.40)
    # Si recortamos, mantenemos total razonable.
    return round(home_xg, 3), round(away_xg, 3), {"p_home_market": ph, "p_draw_market": pd, "p_away_market": pa, "total_goals_est": total_goals}


def risk_level(prob: float) -> str:
    if prob >= 0.72:
        return "Bajo"
    if prob >= 0.56:
        return "Medio"
    return "Alto"


def stake_suggested(prob: float, value: Optional[float], bank: float, conservative: bool = True) -> float:
    # Regla simple: no usar martingala. Ajusta por confianza y valor.
    if bank <= 0:
        return 0.0
    if value is not None and value < -0.03:
        return 0.0
    factor = 0.05
    if prob >= 0.72:
        factor = 0.15 if not conservative else 0.10
    elif prob >= 0.56:
        factor = 0.10 if not conservative else 0.06
    else:
        factor = 0.05 if not conservative else 0.03
    if value is not None and value > 0.06:
        factor += 0.03
    return round(max(0.0, min(bank * factor, bank * 0.25)), 2)


def market_value(prob: float, odds: Optional[float]) -> Optional[float]:
    pi = implied_probability(odds)
    if pi is None:
        return None
    return prob - pi


def decision_from_value(prob: float, odds: Optional[float]) -> str:
    v = market_value(prob, odds)
    if v is None:
        return "Sin cuota para comparar"
    if v >= 0.06:
        return "✅ Valor claro"
    if v > 0:
        return "🟡 Valor pequeño"
    if prob >= 0.72:
        return "🟦 Alta prob., cuota ajustada"
    return "❌ Sin valor"


def simulate_match(
    home_team: str,
    away_team: str,
    home_xg: float,
    away_xg: float,
    simulations: int = 50000,
    bank: float = 3.0,
    odds: Optional[Dict[str, Optional[float]]] = None,
    seed: Optional[int] = None,
) -> PredictionResult:
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(home_xg, simulations)
    away_goals = rng.poisson(away_xg, simulations)
    total_goals = home_goals + away_goals

    p_home = float(np.mean(home_goals > away_goals))
    p_draw = float(np.mean(home_goals == away_goals))
    p_away = float(np.mean(home_goals < away_goals))
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    p_12 = p_home + p_away
    p_over15 = float(np.mean(total_goals > 1.5))
    p_over25 = float(np.mean(total_goals > 2.5))
    p_under35 = float(np.mean(total_goals < 3.5))
    p_btts = float(np.mean((home_goals > 0) & (away_goals > 0)))

    scores = Counter(zip(home_goals.tolist(), away_goals.tolist())).most_common(10)
    top_scores = [(f"{home_team} {h}-{a} {away_team}", c / simulations) for (h, a), c in scores]

    odds = odds or {}
    markets = [
        (f"Gana {home_team}", p_home, odds.get("home")),
        ("Empate", p_draw, odds.get("draw")),
        (f"Gana {away_team}", p_away, odds.get("away")),
        (f"{home_team} o empate (1X)", p_1x, odds.get("1x")),
        (f"{away_team} o empate (X2)", p_x2, odds.get("x2")),
        ("Más de 1.5 goles", p_over15, odds.get("over15")),
        ("Más de 2.5 goles", p_over25, odds.get("over25")),
        ("Menos de 3.5 goles", p_under35, odds.get("under35")),
        ("Ambos marcan", p_btts, odds.get("btts")),
    ]

    table = []
    for name, prob, market_odds in markets:
        value = market_value(prob, market_odds)
        table.append({
            "Mercado": name,
            "Prob. bot": round(prob * 100, 2),
            "Cuota": market_odds,
            "Prob. cuota": round((implied_probability(market_odds) or 0) * 100, 2) if market_odds else None,
            "Diferencia": round(value * 100, 2) if value is not None else None,
            "Riesgo": risk_level(prob),
            "Stake sugerido": stake_suggested(prob, value, bank),
            "Decisión": decision_from_value(prob, market_odds),
        })

    def score_row(row: Dict[str, object]) -> float:
        prob = float(row["Prob. bot"]) / 100
        diff = row.get("Diferencia")
        val = (float(diff) / 100) if diff is not None else 0
        # Equilibrio: valor + probabilidad. Penaliza picks de menos del 50%.
        return prob * 0.7 + max(val, 0) * 1.4 - (0.12 if prob < 0.50 else 0)

    ranked = sorted(table, key=score_row, reverse=True)
    best_pick = ranked[0]
    # Para ticket buscamos alta probabilidad aunque cuota sea baja.
    ticket_candidates = [r for r in table if float(r["Prob. bot"]) >= 62]
    ticket_pick = sorted(ticket_candidates or table, key=lambda r: float(r["Prob. bot"]), reverse=True)[0]

    return PredictionResult(
        home_team=home_team,
        away_team=away_team,
        home_xg=home_xg,
        away_xg=away_xg,
        simulations=simulations,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_1x=p_1x,
        p_x2=p_x2,
        p_12=p_12,
        p_over15=p_over15,
        p_over25=p_over25,
        p_under35=p_under35,
        p_btts=p_btts,
        top_scores=top_scores,
        market_table=table,
        best_pick=best_pick,
        ticket_pick=ticket_pick,
    )


def result_summary_df(result: PredictionResult) -> pd.DataFrame:
    return pd.DataFrame([
        {"Resultado": f"Gana {result.home_team}", "Probabilidad": result.p_home},
        {"Resultado": "Empate", "Probabilidad": result.p_draw},
        {"Resultado": f"Gana {result.away_team}", "Probabilidad": result.p_away},
        {"Resultado": f"{result.home_team} o empate", "Probabilidad": result.p_1x},
        {"Resultado": f"{result.away_team} o empate", "Probabilidad": result.p_x2},
        {"Resultado": "Más de 1.5 goles", "Probabilidad": result.p_over15},
        {"Resultado": "Más de 2.5 goles", "Probabilidad": result.p_over25},
        {"Resultado": "Menos de 3.5 goles", "Probabilidad": result.p_under35},
        {"Resultado": "Ambos marcan", "Probabilidad": result.p_btts},
    ])
