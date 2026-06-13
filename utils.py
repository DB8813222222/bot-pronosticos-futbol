from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional


def utc_iso_range(days_ahead: int = 3):
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=days_ahead)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def date_yyyy_mm_dd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_money(x: float) -> str:
    return f"${x:.2f}"


def decimal_odds_product(odds):
    result = 1.0
    for odd in odds:
        if odd and float(odd) > 1:
            result *= float(odd)
    return result


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default
