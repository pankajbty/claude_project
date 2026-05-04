"""HTTP front door for the Portfolio Management tool.

Layering
--------
* :mod:`app`        -- domain engine: fetch, score, recommend (unchanged)
* :mod:`portfolio`  -- persistence + P&L evaluation
* this module       -- HTTP routing, parallel universe scan, payload shaping

The handler stays deliberately thin: every endpoint resolves to a single
service call and a JSON response. Business logic does not live here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from jinja2 import Environment, FileSystemLoader

from app import (
    DEFAULT_SCAN_SYMBOLS,
    analyze_symbol,
    build_candidate_rows,
    build_price_points,
    build_report_context,
    format_inr,
)
from portfolio import (
    PortfolioStore,
    evaluate_portfolio,
    normalize_symbol,
)


BASE_DIR = Path(__file__).resolve().parent
PORT = 8010
SCAN_WORKERS = 8
TOP_N = 10

_jinja = Environment(loader=FileSystemLoader(str(BASE_DIR)))
_store = PortfolioStore(BASE_DIR / "portfolio.json")


# ---------------------------------------------------------------------------
# Analysis & universe scan
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def cached_analysis(symbol: str) -> dict:
    return analyze_symbol(normalize_symbol(symbol))


def scan_universe(symbols=None) -> tuple[list[dict], list[str]]:
    """Analyze the full symbol universe in parallel.

    yfinance calls are I/O-bound, so a thread pool gives a multi-x speedup
    on cold cache without changing the engine. ``lru_cache`` then makes
    warm requests effectively free.
    """
    symbols = list(symbols or DEFAULT_SCAN_SYMBOLS)
    candidates: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        future_to_symbol = {
            pool.submit(cached_analysis, sym): sym for sym in symbols
        }
        for future in as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                candidates.append(future.result())
            except Exception as exc:
                errors.append(f"{sym}: {exc}")

    if not candidates:
        raise RuntimeError(
            "No stocks could be scanned. Check symbols or network access."
        )
    return candidates, errors


# ---------------------------------------------------------------------------
# Service layer (pure functions; no HTTP awareness)
# ---------------------------------------------------------------------------

def _buy_rank(item: dict) -> float:
    return item["buy_confidence"] + item["trade"]["decision_score"] * 8


def _sell_score(item: dict) -> float:
    """Heuristic that surfaces names worth trimming.

    Penalises negative trends, rich PE, weak ROE, low risk score, and a
    negative model decision score. Higher is more sell-worthy.
    """
    trade = item["trade"]
    score = item["score"]
    data = item["data"]
    screen = item["buy_screen"]
    pe = data["pe"] or 0
    roe = data["roe"] or 0
    penalty = 0.0
    penalty += abs(min(screen["trend_3m"], 0)) * 0.7
    penalty += abs(min(screen["trend_12m"], 0)) * 0.5
    penalty += max(pe - 45, 0) * 0.4
    penalty += max(12 - roe, 0) * 1.2
    penalty += max(70 - score["score"], 0) * 0.8
    penalty += abs(min(trade["decision_score"], 0)) * 8
    return round(penalty, 1)


def top_buy_candidates(limit: int = TOP_N) -> tuple[list[dict], list[dict], list[str]]:
    universe, errors = scan_universe()
    qualifying = [item for item in universe if item["buy_screen"]["passes"]]
    qualifying.sort(key=_buy_rank, reverse=True)
    return qualifying[:limit], universe, errors


def top_sell_candidates(limit: int = TOP_N) -> tuple[list[dict], list[dict], list[str]]:
    universe, errors = scan_universe()
    scored = [{**item, "sell_score": _sell_score(item)} for item in universe]
    scored.sort(key=lambda item: item["sell_score"], reverse=True)
    sell_only = [
        item for item in scored
        if item["trade"]["action"] == "SELL"
        or item["sell_score"] >= 28
        or item["trade"]["decision_score"] <= -2
    ]
    final = sell_only or scored
    return final[:limit], scored, errors


def best_overall(limit: int = TOP_N) -> tuple[list[dict], list[str]]:
    universe, errors = scan_universe()
    ranked = sorted(universe, key=lambda item: item["rank_score"], reverse=True)
    return ranked[:limit], errors


def portfolio_snapshot() -> dict:
    return evaluate_portfolio(_store.list(), cached_analysis)


# ---------------------------------------------------------------------------
# Payload shaping (presentation)
# ---------------------------------------------------------------------------

def stock_payload(item: dict) -> dict:
    data = item["data"]
    score = item["score"]
    trade = item["trade"]
    report = build_report_context(data, score, trade)

    return {
        "symbol": data["symbol"],
        "name": data["name"],
        "price": data["price"],
        "price_as_of": data["price_as_of"],
        "sector": data["sector"],
        "market_cap": format_inr(data["market_cap"]),
        "revenue": format_inr(data["revenue"]),
        "profit": format_inr(data["profit"]),
        "debt": format_inr(data["debt"]),
        "pe": round(data["pe"] or 0, 2),
        "roe": round(data["roe"] or 0, 2),
        "beta": data["beta"] or 0,
        "risk_score": score["score"],
        "risk_label": score["risk_label"],
        "action": trade["action"],
        "action_class": trade["class"],
        "stance": trade["stance"],
        "confidence": item.get("buy_confidence", 0),
        "confidence_label": item.get("buy_confidence_label", "N/A"),
        "model_score": trade["decision_score"],
        "trend_3m": trade["three_month_return"],
        "trend_12m": trade["one_year_return"],
        "debt_to_revenue": trade["debt_to_revenue"],
        "reasons": trade["reasons"],
        "watchouts": trade["watchouts"],
        "price_points": build_price_points(data["prices"]),
        "quarterly_rows": data["quarterly_rows"],
        "bottom_line": report["bottom_line_text"],
    }


def row_payload(item: dict) -> dict:
    row = build_candidate_rows([item])[0]
    data = item["data"]
    row.update({
        "sector": data["sector"],
        "price_as_of": data["price_as_of"],
        "confidence_label": item.get("buy_confidence_label", "N/A"),
        "company": data["name"],
    })
    return row


def sell_row_payload(item: dict) -> dict:
    return {**row_payload(item), "sell_score": item["sell_score"]}


def overview_payload() -> dict:
    universe, scan_errors = scan_universe()

    ranked = sorted(universe, key=lambda item: item["rank_score"], reverse=True)
    best = ranked[0]

    buy_qualifying = [item for item in universe if item["buy_screen"]["passes"]]
    buy_qualifying.sort(key=_buy_rank, reverse=True)
    best_buy = buy_qualifying[0] if buy_qualifying else best

    scored = [{**item, "sell_score": _sell_score(item)} for item in universe]
    scored.sort(key=lambda item: item["sell_score"], reverse=True)
    sell_pool = [
        item for item in scored
        if item["trade"]["action"] == "SELL"
        or item["sell_score"] >= 28
        or item["trade"]["decision_score"] <= -2
    ] or scored
    best_sell = sell_pool[0] if sell_pool else None

    portfolio = portfolio_snapshot()

    return {
        "best": stock_payload(best),
        "best_buy": stock_payload(best_buy),
        "best_sell": stock_payload(best_sell) if best_sell else None,
        "top_buys": [row_payload(item) for item in buy_qualifying[:TOP_N]],
        "top_sells": [sell_row_payload(item) for item in sell_pool[:TOP_N]],
        "best_candidates": [row_payload(item) for item in ranked[:TOP_N]],
        "portfolio": portfolio,
        "scanned_count": len(universe),
        "errors": scan_errors,
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def json_response(handler, payload, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _send(handler, status, body, "application/json; charset=utf-8")


def html_response(handler, html: str, status: int = 200) -> None:
    _send(handler, status, html.encode("utf-8"), "text/html; charset=utf-8")


def error_response(handler, message: str, status: int = 500) -> None:
    json_response(handler, {"error": message}, status=status)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    if not raw.strip():
        return {}
    return json.loads(raw)


def _query_int(query: dict, key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


class PortfolioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return

    # -- GET --

    def do_GET(self):  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/":
                html = _jinja.get_template("portfolio_app.html").render()
                html_response(self, html)
                return

            if path == "/api/overview":
                json_response(self, overview_payload())
                return

            if path == "/api/buys":
                limit = _query_int(query, "limit", TOP_N)
                rows, _, errors = top_buy_candidates(limit)
                json_response(self, {
                    "rows": [row_payload(item) for item in rows],
                    "errors": errors,
                })
                return

            if path == "/api/sells":
                limit = _query_int(query, "limit", TOP_N)
                rows, _, errors = top_sell_candidates(limit)
                json_response(self, {
                    "rows": [sell_row_payload(item) for item in rows],
                    "errors": errors,
                })
                return

            if path == "/api/scan":
                # Backwards-compatible: existing /api/scan?mode=buy|sell|best
                mode = query.get("mode", ["buy"])[0]
                if mode == "best":
                    rows, errors = best_overall(TOP_N + 2)
                    json_response(self, {
                        "mode": mode,
                        "rows": [row_payload(item) for item in rows],
                        "errors": errors,
                    })
                    return
                if mode == "sell":
                    rows, _, errors = top_sell_candidates(TOP_N + 2)
                    json_response(self, {
                        "mode": mode,
                        "rows": [sell_row_payload(item) for item in rows],
                        "errors": errors,
                    })
                    return
                rows, _, errors = top_buy_candidates(TOP_N + 2)
                json_response(self, {
                    "mode": "buy",
                    "rows": [row_payload(item) for item in rows],
                    "errors": errors,
                })
                return

            if path.startswith("/api/stock/"):
                symbol = unquote(path.replace("/api/stock/", "", 1))
                json_response(self, stock_payload(cached_analysis(symbol)))
                return

            if path == "/api/portfolio":
                json_response(self, portfolio_snapshot())
                return

            error_response(self, "Not found", status=404)
        except ValueError as exc:
            error_response(self, str(exc), status=400)
        except Exception as exc:
            error_response(self, str(exc), status=500)

    # -- POST --

    def do_POST(self):  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/portfolio":
                body = _read_json_body(self)
                holding = _store.add(
                    symbol=body.get("symbol", ""),
                    quantity=body.get("quantity"),
                    avg_price=body.get("avg_price"),
                )
                json_response(self, {
                    "added": holding.to_dict(),
                    "portfolio": portfolio_snapshot(),
                }, status=201)
                return

            error_response(self, "Not found", status=404)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            error_response(self, str(exc), status=400)
        except Exception as exc:
            error_response(self, str(exc), status=500)

    # -- DELETE --

    def do_DELETE(self):  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/portfolio/"):
                symbol = unquote(path.replace("/api/portfolio/", "", 1))
                removed = _store.remove(symbol)
                if not removed:
                    error_response(self, f"{symbol} not in portfolio", status=404)
                    return
                json_response(self, {
                    "removed": symbol,
                    "portfolio": portfolio_snapshot(),
                })
                return

            if path == "/api/portfolio":
                _store.clear()
                json_response(self, portfolio_snapshot())
                return

            error_response(self, "Not found", status=404)
        except ValueError as exc:
            error_response(self, str(exc), status=400)
        except Exception as exc:
            error_response(self, str(exc), status=500)


def run_server(port: int = PORT) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), PortfolioHandler)
    print(f"Portfolio management app running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server(port=PORT)
