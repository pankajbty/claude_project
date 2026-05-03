import json
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


BASE_DIR = Path(__file__).resolve().parent
PORT = 8000


@lru_cache(maxsize=128)
def cached_analysis(symbol):
    normalized = normalize_symbol(symbol)
    return analyze_symbol(normalized)


def normalize_symbol(symbol):
    clean = symbol.strip().upper()
    if not clean:
        return "RELIANCE.NS"
    if "." not in clean:
        return f"{clean}.NS"
    return clean


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler, html, status=200):
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler, message, status=500):
    json_response(handler, {"error": message}, status=status)


def stock_payload(item):
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


def scan_all_cached(symbols=None):
    candidates = []
    errors = []
    for symbol in symbols or DEFAULT_SCAN_SYMBOLS:
        try:
            candidates.append(cached_analysis(symbol.strip().upper()))
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    if not candidates:
        raise RuntimeError("No stocks could be scanned. Check symbols or network access.")
    return candidates, errors


def row_payload(item):
    row = build_candidate_rows([item])[0]
    data = item["data"]
    row.update({
        "sector": data["sector"],
        "price_as_of": data["price_as_of"],
        "confidence_label": item.get("buy_confidence_label", "N/A"),
        "company": data["name"],
    })
    return row


def sell_score(item):
    trade = item["trade"]
    score = item["score"]
    data = item["data"]
    screen = item["buy_screen"]
    pe = data["pe"] or 0
    roe = data["roe"] or 0
    penalty = 0
    penalty += abs(min(screen["trend_3m"], 0)) * 0.7
    penalty += abs(min(screen["trend_12m"], 0)) * 0.5
    penalty += max(pe - 45, 0) * 0.4
    penalty += max(12 - roe, 0) * 1.2
    penalty += max(70 - score["score"], 0) * 0.8
    penalty += abs(min(trade["decision_score"], 0)) * 8
    return round(penalty, 1)


def scan_sell_candidates(symbols=None):
    raw_candidates, errors = scan_all_cached(symbols)
    candidates = []
    for item in raw_candidates:
        scored = dict(item)
        scored["sell_score"] = sell_score(scored)
        candidates.append(scored)

    candidates.sort(key=lambda item: item["sell_score"], reverse=True)
    sell_candidates = [
        item for item in candidates
        if item["trade"]["action"] == "SELL"
        or item["sell_score"] >= 28
        or item["trade"]["decision_score"] <= -2
    ]
    if not sell_candidates:
        sell_candidates = candidates[:10]
    return sell_candidates[:10], candidates, errors


def overview_payload():
    all_candidates, scan_errors = scan_all_cached(DEFAULT_SCAN_SYMBOLS)
    ranked = sorted(all_candidates, key=lambda item: item["rank_score"], reverse=True)
    best = ranked[0]
    buy_candidates = [item for item in all_candidates if item["buy_screen"]["passes"]]
    buy_candidates.sort(
        key=lambda item: item["buy_confidence"] + item["trade"]["decision_score"] * 8,
        reverse=True,
    )
    buy_winner = buy_candidates[0] if buy_candidates else best
    sell_candidates, _, sell_errors = scan_sell_candidates(DEFAULT_SCAN_SYMBOLS)
    sell_winner = sell_candidates[0] if sell_candidates else None

    return {
        "best": stock_payload(best),
        "best_buy": stock_payload(buy_winner),
        "best_sell": stock_payload(sell_winner) if sell_winner else None,
        "buy_candidates": [row_payload(item) for item in buy_candidates[:8]],
        "best_candidates": [row_payload(item) for item in ranked[:8]],
        "sell_candidates": [
            {**row_payload(item), "sell_score": item["sell_score"]}
            for item in sell_candidates[:8]
        ],
        "scanned_count": len(all_candidates),
        "errors": scan_errors + sell_errors,
    }


class PortfolioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/":
                env = Environment(loader=FileSystemLoader(str(BASE_DIR)))
                html = env.get_template("portfolio_app.html").render()
                html_response(self, html)
                return

            if path == "/api/overview":
                json_response(self, overview_payload())
                return

            if path.startswith("/api/stock/"):
                symbol = unquote(path.replace("/api/stock/", "", 1))
                json_response(self, stock_payload(cached_analysis(symbol)))
                return

            if path == "/api/scan":
                mode = query.get("mode", ["buy"])[0]
                all_candidates, errors = scan_all_cached(DEFAULT_SCAN_SYMBOLS)
                if mode == "best":
                    candidates = sorted(all_candidates, key=lambda item: item["rank_score"], reverse=True)
                    json_response(self, {
                        "mode": mode,
                        "rows": [row_payload(item) for item in candidates[:12]],
                        "errors": errors,
                    })
                    return
                if mode == "sell":
                    sell_candidates, _, errors = scan_sell_candidates(DEFAULT_SCAN_SYMBOLS)
                    json_response(self, {
                        "mode": mode,
                        "rows": [
                            {**row_payload(item), "sell_score": item["sell_score"]}
                            for item in sell_candidates[:12]
                        ],
                        "errors": errors,
                    })
                    return

                buy_candidates = [item for item in all_candidates if item["buy_screen"]["passes"]]
                buy_candidates.sort(
                    key=lambda item: item["buy_confidence"] + item["trade"]["decision_score"] * 8,
                    reverse=True,
                )
                json_response(self, {
                    "mode": "buy",
                    "rows": [row_payload(item) for item in buy_candidates[:12]],
                    "errors": errors,
                })
                return

            error_response(self, "Not found", status=404)
        except Exception as exc:
            error_response(self, str(exc), status=500)


def run_server(port=PORT):
    server = ThreadingHTTPServer(("127.0.0.1", port), PortfolioHandler)
    print(f"Portfolio management app running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
