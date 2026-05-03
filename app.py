import yfinance as yf
import pandas as pd
from jinja2 import Environment, FileSystemLoader


DEFAULT_SCAN_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "LT.NS", "BHARTIARTL.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "ITC.NS", "HINDUNILVR.NS", "MARUTI.NS", "M&M.NS", "TITAN.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "POWERGRID.NS", "BEL.NS", "HAL.NS", "COCHINSHIP.NS",
    "IRCTC.NS", "TRENT.NS", "DLF.NS", "JIOFIN.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "TECHM.NS", "WIPRO.NS", "PERSISTENT.NS",
    "EICHERMOT.NS", "TATASTEEL.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS",
    "APOLLOHOSP.NS", "DRREDDY.NS", "LUPIN.NS", "ZYDUSLIFE.NS",
    "ONGC.NS", "COALINDIA.NS", "GAIL.NS", "IOC.NS", "BPCL.NS",
    "ADANIPORTS.NS", "INDIGO.NS", "NAUKRI.NS", "POLYCAB.NS",
    "PIDILITIND.NS", "ASIANPAINT.NS", "DMART.NS", "VBL.NS",
]

BUY_SCREEN = {
    "max_pe": 45,
    "min_roe": 12,
    "min_risk_score": 68,
    "min_decision_score": 3,
    "min_3m_return": -15,
    "min_12m_return": -20,
    "max_debt_to_revenue": 1.2,
}


def format_inr(value):
    if not value:
        return "N/A"

    crore = value / 10_000_000
    if abs(crore) >= 1000:
        return f"₹{crore / 1000:,.2f}K Cr"
    return f"₹{crore:,.0f} Cr"


def build_price_points(prices, width=300, height=200, padding=20):
    clean_prices = [float(price) for price in prices if pd.notna(price)]
    if len(clean_prices) < 2:
        return "0,100 300,100"

    min_price = min(clean_prices)
    max_price = max(clean_prices)
    spread = max(max_price - min_price, 1)
    step = width / (len(clean_prices) - 1)

    points = []
    for index, price in enumerate(clean_prices):
        x = index * step
        y = height - padding - ((price - min_price) / spread * (height - padding * 2))
        points.append(f"{x:.1f},{y:.1f}")

    return " ".join(points)


def clean_price_list(prices):
    return [float(price) for price in prices if pd.notna(price)]


def percent_change(start, end):
    if not start:
        return 0
    return ((end - start) / start) * 100


def format_percent(value):
    return f"{value:+.1f}%"


def latest_market_price(info, hist):
    if hist is not None and not hist.empty and "Close" in hist:
        close_prices = hist["Close"].dropna()
        if not close_prices.empty:
            return round(float(close_prices.iloc[-1]), 2)

    candidates = [
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("previousClose"),
    ]

    for candidate in candidates:
        if candidate:
            return round(float(candidate), 2)

    return 0


def latest_price_timestamp(hist):
    if hist is None or hist.empty:
        return "Latest traded quote"

    latest_index = hist.index[-1]
    try:
        return pd.to_datetime(latest_index).strftime("%d %b %Y")
    except Exception:
        return "Latest traded quote"


def risk_label(score):
    if score >= 75:
        return "High Conviction / Lower Risk"
    if score >= 60:
        return "Moderate Risk"
    if score >= 45:
        return "Moderate-High Risk"
    return "High Risk"


def signal_class(value, good_threshold=70, warn_threshold=55):
    if value >= good_threshold:
        return "green"
    if value >= warn_threshold:
        return "amber"
    return "red"


def recommendation_class(action):
    if action == "BUY":
        return "green"
    if action == "SELL":
        return "red"
    return "amber"


def build_trade_recommendation(data, score):
    prices = clean_price_list(data["prices"])
    latest = prices[-1] if prices else data["price"]
    one_year_return = percent_change(prices[0], latest) if len(prices) > 1 else 0
    three_month_return = percent_change(prices[-64], latest) if len(prices) > 64 else one_year_return
    debt_to_revenue = data["debt"] / data["revenue"] if data["revenue"] else None

    decision_score = 0
    reasons = []
    watchouts = []

    pe = data["pe"] or 0
    roe = data["roe"] or 0

    if pe <= 0:
        watchouts.append("PE is unavailable, so valuation confidence is lower.")
    elif pe < 25:
        decision_score += 2
        reasons.append(f"Valuation is reasonable with PE near {pe:.1f}.")
    elif pe < 40:
        decision_score += 1
        reasons.append(f"PE near {pe:.1f} is acceptable if growth continues.")
    elif pe > 60:
        decision_score -= 2
        watchouts.append(f"PE near {pe:.1f} is rich and leaves less margin of safety.")
    else:
        decision_score -= 1
        watchouts.append(f"PE near {pe:.1f} looks elevated versus a conservative entry.")

    if roe > 20:
        decision_score += 2
        reasons.append(f"ROE around {roe:.1f}% shows strong capital efficiency.")
    elif roe > 15:
        decision_score += 1
        reasons.append(f"ROE around {roe:.1f}% is healthy.")
    elif roe > 0:
        decision_score -= 1
        watchouts.append(f"ROE around {roe:.1f}% is not strong enough to fully offset valuation risk.")
    else:
        decision_score -= 1
        watchouts.append("ROE is unavailable or weak in the fetched data.")

    if debt_to_revenue is not None:
        if debt_to_revenue < 0.3:
            decision_score += 1
            reasons.append("Debt looks manageable relative to revenue.")
        elif debt_to_revenue > 1:
            decision_score -= 1
            watchouts.append("Debt is high relative to revenue.")

    if three_month_return > 12:
        decision_score += 1
        reasons.append(f"Short-term trend is positive at {format_percent(three_month_return)} over roughly 3 months.")
    elif three_month_return > 0:
        reasons.append(f"Short-term trend is mildly positive at {format_percent(three_month_return)} over roughly 3 months.")
    elif three_month_return < -20:
        decision_score -= 2
        watchouts.append(f"Short-term trend is sharply weak at {format_percent(three_month_return)} over roughly 3 months.")
    elif three_month_return < -12:
        decision_score -= 1
        watchouts.append(f"Short-term trend is weak at {format_percent(three_month_return)} over roughly 3 months.")

    if one_year_return > 25:
        decision_score += 1
        reasons.append(f"12M price momentum is strong at {format_percent(one_year_return)}.")
    elif one_year_return < -20:
        decision_score -= 1
        watchouts.append(f"12M price momentum is negative at {format_percent(one_year_return)}.")

    if score["score"] >= 70:
        decision_score += 1
        reasons.append("Composite risk score supports a constructive stance.")
    elif score["score"] < 55:
        decision_score -= 1
        watchouts.append("Composite risk score is below the preferred threshold.")

    if decision_score >= 4:
        action = "BUY"
        stance = "Fresh entry can be considered in phases."
        confidence = "Medium"
    elif decision_score <= -3:
        action = "SELL"
        stance = "Avoid fresh buying; existing holders can reduce on strength."
        confidence = "Medium"
    else:
        action = "HOLD"
        stance = "Wait for better valuation, stronger earnings, or a cleaner technical setup."
        confidence = "Medium-Low" if abs(decision_score) <= 1 else "Medium"

    if action == "BUY" and three_month_return < -20:
        action = "HOLD"
        stance = "Fundamentals screen well, but wait for price stabilization before fresh buying."
        confidence = "Medium"

    if not reasons:
        reasons.append("No strong bullish factor is dominant from the fetched data.")
    if not watchouts:
        watchouts.append("Main risk is normal market volatility and data freshness.")

    return {
        "action": action,
        "stance": stance,
        "confidence": confidence,
        "class": recommendation_class(action),
        "decision_score": decision_score,
        "one_year_return": format_percent(one_year_return),
        "three_month_return": format_percent(three_month_return),
        "debt_to_revenue": f"{debt_to_revenue:.2f}x" if debt_to_revenue is not None else "N/A",
        "reasons": reasons[:4],
        "watchouts": watchouts[:4],
    }


def parse_percent(value):
    if isinstance(value, str):
        return float(value.replace("%", "").replace("+", ""))
    return float(value)


def parse_ratio(value):
    if value == "N/A":
        return None
    if isinstance(value, str):
        return float(value.replace("x", ""))
    return float(value)


def build_buy_screen_result(data, score, trade):
    pe = data["pe"] or 0
    roe = data["roe"] or 0
    trend_3m = parse_percent(trade["three_month_return"])
    trend_12m = parse_percent(trade["one_year_return"])
    debt_to_revenue = parse_ratio(trade["debt_to_revenue"])

    checks = [
        ("Recommendation", trade["action"] == "BUY", f"Model action is {trade['action']}."),
        ("Valuation", 0 < pe <= BUY_SCREEN["max_pe"], f"PE {pe:.1f} should be <= {BUY_SCREEN['max_pe']}."),
        ("ROE", roe >= BUY_SCREEN["min_roe"], f"ROE {roe:.1f}% should be >= {BUY_SCREEN['min_roe']}%."),
        ("Risk score", score["score"] >= BUY_SCREEN["min_risk_score"], f"Risk score {score['score']} should be >= {BUY_SCREEN['min_risk_score']}."),
        ("Model score", trade["decision_score"] >= BUY_SCREEN["min_decision_score"], f"Model score {trade['decision_score']} should be >= {BUY_SCREEN['min_decision_score']}."),
        ("3M trend", trend_3m >= BUY_SCREEN["min_3m_return"], f"3M trend {format_percent(trend_3m)} should be >= {BUY_SCREEN['min_3m_return']}%."),
        ("12M trend", trend_12m >= BUY_SCREEN["min_12m_return"], f"12M trend {format_percent(trend_12m)} should be >= {BUY_SCREEN['min_12m_return']}%."),
    ]

    if debt_to_revenue is not None:
        checks.append((
            "Leverage",
            debt_to_revenue <= BUY_SCREEN["max_debt_to_revenue"],
            f"Debt/revenue {debt_to_revenue:.2f}x should be <= {BUY_SCREEN['max_debt_to_revenue']}x.",
        ))

    passed = [message for _, ok, message in checks if ok]
    failed = [message for _, ok, message in checks if not ok]

    return {
        "passes": not failed,
        "passed": passed,
        "failed": failed,
        "trend_3m": trend_3m,
        "trend_12m": trend_12m,
        "debt_to_revenue": debt_to_revenue,
    }


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def calculate_buy_confidence(data, score, trade, buy_screen):
    pe = data["pe"] or 0
    roe = data["roe"] or 0
    debt_to_revenue = buy_screen["debt_to_revenue"]
    trend_3m = buy_screen["trend_3m"]
    trend_12m = buy_screen["trend_12m"]

    confidence = 50
    confidence += clamp(trade["decision_score"], -5, 10) * 4
    confidence += (score["score"] - 60) * 0.6

    if 0 < pe <= 20:
        confidence += 12
    elif pe <= 30:
        confidence += 8
    elif pe <= 45:
        confidence += 3
    elif pe > 60:
        confidence -= 12

    if roe >= 25:
        confidence += 12
    elif roe >= 18:
        confidence += 8
    elif roe >= 12:
        confidence += 4
    elif roe > 0:
        confidence -= 6

    if debt_to_revenue is not None:
        if debt_to_revenue <= 0.3:
            confidence += 8
        elif debt_to_revenue <= 0.8:
            confidence += 3
        elif debt_to_revenue > 1.2:
            confidence -= 10

    if trend_3m >= 12:
        confidence += 8
    elif trend_3m >= 0:
        confidence += 3
    elif trend_3m < -15:
        confidence -= 8
    elif trend_3m < -10:
        confidence -= 6
    else:
        confidence -= 2

    if trend_12m >= 20:
        confidence += 8
    elif trend_12m >= 0:
        confidence += 3
    elif trend_12m < -20:
        confidence -= 8
    elif trend_12m < -10:
        confidence -= 5
    else:
        confidence -= 2

    if not buy_screen["passes"]:
        confidence -= len(buy_screen["failed"]) * 6

    return int(round(clamp(confidence, 0, 95)))


def confidence_label(confidence):
    if confidence >= 80:
        return "High"
    if confidence >= 65:
        return "Medium-High"
    if confidence >= 50:
        return "Medium"
    return "Low"


def enrich_analysis(data):
    score = calculate_risk(data)
    score["risk_label"] = risk_label(score["score"])
    score["valuation_class"] = "red" if score["val_signal"] == "Expensive" else "green"
    score["financial_class"] = signal_class(score["financial"])
    score["growth_class"] = signal_class(score["growth"])
    score["bottom_line_class"] = signal_class(score["score"], 75, 55)
    trade = build_trade_recommendation(data, score)
    return score, trade


def build_report_context(data, score, trade, scanned_count=None):
    scan_note = ""
    if scanned_count:
        scan_note = f" Selected as the strongest candidate from {scanned_count} scanned NSE symbols."

    report = {
        "price_points": build_price_points(data["prices"]),
        "revenue_display": format_inr(data["revenue"]),
        "profit_display": format_inr(data["profit"]),
        "debt_display": format_inr(data["debt"]),
        "market_cap_display": format_inr(data["market_cap"]),
        "subtitle": f"{data['sector']} | Beta {data['beta'] or 'N/A'}",
        "latest_update": (
            f"{data['name']} shows {score['val_signal'].lower()} valuation signals and "
            f"{score['fin_signal'].lower()} financial quality. Momentum should be weighed "
            f"against earnings volatility, execution risk, and market-wide drawdowns.{scan_note}"
        ),
        "bottom_line": (
            f"Action: {trade['action']} | Data stance: {data['recommendation'].upper()}"
        ),
        "bottom_line_text": (
            f"{trade['stance']} This is an analytical signal, not personalized financial advice; "
            "position sizing should depend on your risk tolerance, time horizon, and portfolio exposure."
        ),
        "catalysts": [
            "Improving order inflow and revenue visibility",
            "Sector capex and policy tailwinds",
            "Operating leverage from larger execution scale"
        ],
        "risks": [
            "Rich valuation or momentum reversal",
            "Quarterly earnings volatility",
            "Execution delays and order concentration"
        ]
    }
    return report


def candidate_rank_score(score, trade):
    action_bonus = {"BUY": 8, "HOLD": 2, "SELL": -8}.get(trade["action"], 0)
    return trade["decision_score"] * 10 + score["score"] + action_bonus


def analyze_symbol(symbol):
    data = fetch_stock(symbol)
    score, trade = enrich_analysis(data)
    buy_screen = build_buy_screen_result(data, score, trade)
    buy_confidence = calculate_buy_confidence(data, score, trade, buy_screen)
    return {
        "symbol": symbol,
        "data": data,
        "score": score,
        "trade": trade,
        "rank_score": candidate_rank_score(score, trade),
        "buy_screen": buy_screen,
        "buy_confidence": buy_confidence,
        "buy_confidence_label": confidence_label(buy_confidence),
    }


def scan_best_stock(symbols=None):
    candidates = []
    errors = []

    for symbol in symbols or DEFAULT_SCAN_SYMBOLS:
        try:
            analysis = analyze_symbol(symbol.strip().upper())
            candidates.append(analysis)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    if not candidates:
        raise RuntimeError("No stocks could be scanned. Check symbols or network access.")

    candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    return candidates[0], candidates, errors


def buy_candidate_rank_score(item):
    trade = item["trade"]
    score = item["score"]
    screen = item["buy_screen"]
    momentum_bonus = max(screen["trend_3m"], 0) * 0.4 + max(screen["trend_12m"], 0) * 0.2
    return item["buy_confidence"] + trade["decision_score"] * 8 + score["score"] * 0.5 + momentum_bonus


def scan_buyable_stocks(symbols=None):
    _, candidates, errors = scan_best_stock(symbols)
    buy_candidates = [item for item in candidates if item["buy_screen"]["passes"]]

    if buy_candidates:
        buy_candidates.sort(key=buy_candidate_rank_score, reverse=True)
        winner = buy_candidates[0]
    else:
        candidates.sort(key=lambda item: len(item["buy_screen"]["failed"]))
        winner = candidates[0]

    return winner, buy_candidates, candidates, errors


def build_candidate_rows(candidates):
    rows = []
    for item in candidates:
        data = item["data"]
        score = item["score"]
        trade = item["trade"]
        rows.append({
            "symbol": data["symbol"],
            "name": data["name"],
            "price": f"₹{data['price']}",
            "action": trade["action"],
            "action_class": trade["class"],
            "model_score": trade["decision_score"],
            "risk_score": score["score"],
            "pe": f"{(data['pe'] or 0):.1f}",
            "roe": f"{(data['roe'] or 0):.1f}%",
            "trend_3m": trade["three_month_return"],
            "trend_12m": trade["one_year_return"],
            "rank_score": item["rank_score"],
            "failed_checks": "; ".join(item.get("buy_screen", {}).get("failed", [])),
            "buy_confidence": item.get("buy_confidence", 0),
            "buy_confidence_label": item.get("buy_confidence_label", "N/A"),
        })
    return rows

# -------------------------
# FETCH DATA
# -------------------------
def fetch_stock(symbol):
    stock = yf.Ticker(symbol)

    info = stock.info
    hist = stock.history(period="1y")
    quarterly_rows = []

    try:
        quarterly = stock.quarterly_income_stmt
        for quarter in quarterly.columns[:4]:
            revenue = quarterly.loc["Total Revenue", quarter] if "Total Revenue" in quarterly.index else 0
            profit = quarterly.loc["Net Income", quarter] if "Net Income" in quarterly.index else 0
            quarterly_rows.append({
                "quarter": pd.to_datetime(quarter).strftime("%b %Y"),
                "revenue": format_inr(revenue),
                "profit": format_inr(profit)
            })
    except Exception:
        quarterly_rows = []

    data = {
        "name": info.get("shortName", symbol),
        "symbol": symbol.replace(".NS", ""),
        "price": latest_market_price(info, hist),
        "price_as_of": latest_price_timestamp(hist),
        "pe": info.get("trailingPE", 0),
        "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
        "revenue": info.get("totalRevenue", 0),
        "profit": info.get("netIncomeToCommon", 0),
        "debt": info.get("totalDebt", 0),
        "market_cap": info.get("marketCap", 0),
        "beta": info.get("beta", 0),
        "sector": info.get("sector", "Market-linked business"),
        "recommendation": info.get("recommendationKey", "hold").replace("_", " ").title(),
        "prices": hist["Close"].tolist(),
        "quarterly_rows": quarterly_rows
    }

    return data


# -------------------------
# RISK CALCULATION
# -------------------------
def calculate_risk(data):

    # Valuation
    if data["pe"] > 40:
        valuation = 50
        val_signal = "Expensive"
    else:
        valuation = 80
        val_signal = "Reasonable"

    # Financials
    if data["roe"] > 15:
        financial = 80
        fin_signal = "Strong"
    else:
        financial = 60
        fin_signal = "Average"

    # Growth proxy (simple)
    growth = 70 if data["revenue"] > 0 else 50

    score = int(valuation*0.35 + financial*0.35 + growth*0.30)

    return {
        "score": score,
        "valuation": valuation,
        "financial": financial,
        "growth": growth,
        "val_signal": val_signal,
        "fin_signal": fin_signal
    }


# -------------------------
# GENERATE HTML
# -------------------------
def generate_report(symbol):

    data = fetch_stock(symbol)
    score, trade = enrich_analysis(data)
    report = build_report_context(data, score, trade)

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("template.html")

    html = template.render(data=data, score=score, report=report, trade=trade)

    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Report generated: report.html")


def generate_best_stock_report(symbols=None):
    winner, candidates, errors = scan_best_stock(symbols)
    data = winner["data"]
    score = winner["score"]
    trade = winner["trade"]
    report = build_report_context(data, score, trade, scanned_count=len(candidates))
    report["scan_errors"] = errors
    report["candidate_rows"] = build_candidate_rows(candidates[:12])

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("template.html")

    html = template.render(
        data=data,
        score=score,
        report=report,
        trade=trade,
        scan={
            "enabled": True,
            "scanned_count": len(candidates),
            "symbols": ", ".join(item["data"]["symbol"] for item in candidates),
        },
    )

    with open("best_stock_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Best stock report generated: best_stock_report.html")
    print(f"🏆 Top candidate: {data['symbol']} ({trade['action']}) at ₹{data['price']}")


def generate_buy_stocks_report(symbols=None):
    winner, buy_candidates, all_candidates, errors = scan_buyable_stocks(symbols)
    data = winner["data"]
    score = winner["score"]
    trade = winner["trade"]

    if buy_candidates:
        candidate_rows = build_candidate_rows(buy_candidates)
        scanned_note = (
            f"Found {len(buy_candidates)} stocks that passed the BUY screen from "
            f"{len(all_candidates)} scanned NSE symbols."
        )
    else:
        candidate_rows = build_candidate_rows(all_candidates[:12])
        scanned_note = (
            "No stocks passed every BUY screen. Showing the nearest candidates so you can "
            "see which checks failed."
        )

    report = build_report_context(data, score, trade, scanned_count=len(all_candidates))
    report["scan_errors"] = errors
    report["candidate_rows"] = candidate_rows
    report["buy_confidence"] = winner["buy_confidence"]
    report["buy_confidence_label"] = winner["buy_confidence_label"]
    report["latest_update"] = (
        f"{scanned_note} BUY screen thresholds: PE <= {BUY_SCREEN['max_pe']}, "
        f"ROE >= {BUY_SCREEN['min_roe']}%, risk score >= {BUY_SCREEN['min_risk_score']}, "
        f"model score >= {BUY_SCREEN['min_decision_score']}, 3M trend >= {BUY_SCREEN['min_3m_return']}%, "
        f"12M trend >= {BUY_SCREEN['min_12m_return']}%, debt/revenue <= {BUY_SCREEN['max_debt_to_revenue']}x."
    )

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("template.html")

    html = template.render(
        data=data,
        score=score,
        report=report,
        trade=trade,
        scan={
            "enabled": True,
            "mode": "BUY",
            "scanned_count": len(all_candidates),
            "passed_count": len(buy_candidates),
            "symbols": ", ".join(item["data"]["symbol"] for item in all_candidates),
        },
    )

    with open("buy_stocks_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Buy stocks report generated: buy_stocks_report.html")
    if buy_candidates:
        print(f"🟢 Buy candidates found: {len(buy_candidates)}")
        print(f"🏆 Best buy candidate: {data['symbol']} at ₹{data['price']}")
    else:
        print("⚠️ No stocks passed all BUY filters. Report shows nearest candidates.")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    stock = input("Enter stock (e.g. COCHINSHIP.NS), BEST, or BUY: ").strip()
    if stock.upper() == "BEST":
        generate_best_stock_report()
    elif stock.upper() == "BUY":
        generate_buy_stocks_report()
    else:
        generate_report(stock)
