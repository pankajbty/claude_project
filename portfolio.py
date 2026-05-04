"""Portfolio persistence and live P&L evaluation.

Holdings live in a single JSON file (``portfolio.json``) so the data is
debuggable and trivial to back up. Writes are atomic (temp file +
``os.replace``) and guarded by a ``threading.Lock`` so the
``ThreadingHTTPServer`` in :mod:`web_app` can safely serve concurrent
requests.

Buying the same symbol twice pools the lots into a single position with a
weighted-average buy price -- this is the only way cost basis (and hence
P&L) stays correct across multiple tranches.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


PORTFOLIO_FILENAME = "portfolio.json"


def normalize_symbol(symbol: str) -> str:
    clean = (symbol or "").strip().upper()
    if not clean:
        raise ValueError("Symbol is required")
    return clean if "." in clean else f"{clean}.NS"


def display_symbol(symbol: str) -> str:
    return symbol.replace(".NS", "")


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: float
    avg_price: float
    added_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Holding":
        return cls(
            symbol=normalize_symbol(payload["symbol"]),
            quantity=float(payload["quantity"]),
            avg_price=float(payload["avg_price"]),
            added_at=payload.get("added_at") or _now_iso(),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortfolioStore:
    """Thread-safe JSON-backed repository for portfolio holdings."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path or PORTFOLIO_FILENAME)
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return list(payload.get("holdings", []))

    def _write(self, holdings: list[dict]) -> None:
        directory = str(self._path.parent or ".")
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=directory, encoding="utf-8"
        ) as tmp:
            json.dump({"holdings": holdings}, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name
        os.replace(tmp_path, self._path)

    def list(self) -> list[Holding]:
        with self._lock:
            return [Holding.from_dict(item) for item in self._read()]

    def add(self, symbol: str, quantity: float, avg_price: float) -> Holding:
        if quantity is None or float(quantity) <= 0:
            raise ValueError("Quantity must be positive")
        if avg_price is None or float(avg_price) <= 0:
            raise ValueError("Average price must be positive")

        normalized = normalize_symbol(symbol)
        quantity = float(quantity)
        avg_price = float(avg_price)

        with self._lock:
            holdings = self._read()
            existing = next(
                (h for h in holdings if h["symbol"] == normalized), None
            )
            if existing:
                total_qty = float(existing["quantity"]) + quantity
                total_cost = (
                    float(existing["quantity"]) * float(existing["avg_price"])
                    + quantity * avg_price
                )
                existing["quantity"] = total_qty
                existing["avg_price"] = total_cost / total_qty
                merged = existing
            else:
                merged = {
                    "symbol": normalized,
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "added_at": _now_iso(),
                }
                holdings.append(merged)
            self._write(holdings)
            return Holding.from_dict(merged)

    def remove(self, symbol: str) -> bool:
        normalized = normalize_symbol(symbol)
        with self._lock:
            holdings = self._read()
            kept = [h for h in holdings if h["symbol"] != normalized]
            if len(kept) == len(holdings):
                return False
            self._write(kept)
            return True

    def clear(self) -> None:
        with self._lock:
            self._write([])


def _portfolio_action_label(action: str, pnl_pct: float) -> str:
    if action == "BUY":
        return "Buy more"
    if action == "SELL":
        return "Trim / exit" if pnl_pct >= 0 else "Cut loss"
    return "Hold"


def evaluate_holding(
    holding: Holding, analyzer: Callable[[str], dict]
) -> dict:
    analysis = analyzer(holding.symbol)
    data = analysis["data"]
    trade = analysis["trade"]
    score = analysis["score"]

    invested = holding.quantity * holding.avg_price
    current_value = holding.quantity * data["price"]
    pnl = current_value - invested
    pnl_pct = (pnl / invested * 100.0) if invested else 0.0
    pnl_class = "green" if pnl >= 0 else "red"

    return {
        "symbol": display_symbol(holding.symbol),
        "raw_symbol": holding.symbol,
        "name": data["name"],
        "quantity": holding.quantity,
        "avg_price": round(holding.avg_price, 2),
        "current_price": data["price"],
        "invested": round(invested, 2),
        "current_value": round(current_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_class": pnl_class,
        "action": trade["action"],
        "action_class": trade["class"],
        "recommendation": _portfolio_action_label(trade["action"], pnl_pct),
        "confidence": analysis.get("buy_confidence", 0),
        "confidence_label": analysis.get("buy_confidence_label", "N/A"),
        "risk_score": score["score"],
        "stance": trade["stance"],
        "added_at": holding.added_at,
    }


def evaluate_portfolio(
    holdings: Iterable[Holding], analyzer: Callable[[str], dict]
) -> dict:
    rows: list[dict] = []
    errors: list[str] = []
    for holding in holdings:
        try:
            rows.append(evaluate_holding(holding, analyzer))
        except Exception as exc:
            errors.append(f"{display_symbol(holding.symbol)}: {exc}")

    invested = sum(row["invested"] for row in rows)
    current_value = sum(row["current_value"] for row in rows)
    pnl = current_value - invested
    pnl_pct = (pnl / invested * 100.0) if invested else 0.0
    winners = sum(1 for row in rows if row["pnl"] > 0)
    losers = sum(1 for row in rows if row["pnl"] < 0)

    return {
        "holdings": rows,
        "totals": {
            "holdings_count": len(rows),
            "invested": round(invested, 2),
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_class": "green" if pnl >= 0 else "red",
            "winners": winners,
            "losers": losers,
        },
        "errors": errors,
    }
