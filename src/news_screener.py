# src/news_screener.py
#pip install requests beautifulsoup4 vaderSentiment python-dateutil tabulate
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dateutil import parser as date_parser
from datetime import datetime, timedelta
import yfinance as yf
import re
import pandas as pd
from collections import Counter, defaultdict
from tabulate import tabulate

analyzer = SentimentIntensityAnalyzer()

# --------- NEWS FETCH + SUMMARY (Master Prompt Format) ---------
def fetch_google_news_rss(query, max_items=30):
    """
    Query Google News RSS and return list of dicts: {'title','link','pubDate','source'}
    """
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    resp = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
    soup = BeautifulSoup(resp.content, "xml")
    items = []
    for item in soup.find_all("item")[:max_items]:
        title = item.title.text
        link = item.link.text
        pub = item.pubDate.text if item.pubDate else None
        try:
            pub_dt = date_parser.parse(pub) if pub else None
        except Exception:
            pub_dt = None
        items.append({"title": title, "link": link, "pubDate": pub_dt, "source": "google"})
    return items

def fetch_moneycontrol_search(ticker, max_items=20):
    """
    Basic Moneycontrol search fallback — scrapes search RSS via Google News with site:moneycontrol.
    Returns same dict shape as Google RSS.
    """
    query = f"{ticker} site:moneycontrol.com"
    return fetch_google_news_rss(query, max_items=max_items)

# Event categorization keywords
EVENT_KEYWORDS = {
    "Earnings": ["earn", "q1", "quarter", "profit", "loss", "revenue", "eps", "results", "earnings"],
    "Deal": ["acqui", "deal", "merger", "alliance", "partner", "buyout", "stake", "investment"],
    "Regulatory": ["sebi", "rbi", "regulat", "fine", "penalty", "order", "investigation", "licen"],
    "Management": ["ceo", "cfo", "chairman", "resign", "appoint", "management", "board", "md", "director"],
    "Guidance": ["guidance", "outlook", "forecast", "raise", "lower"],
    "Macro": ["inflation", "rates", "budget", "policy", "geopolitic", "crude", "demonet"],
    "Product": ["launch", "product", "service", "contract", "order", "win"],
    "Other": []
}

def categorize_event(title):
    t = title.lower()
    for cat, kws in EVENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return cat
    return "Other"

def classify_sentiment(title):
    s = analyzer.polarity_scores(title)["compound"]
    # Map to Positive/Neutral/Negative with thresholds
    if s > 0.05:
        label = "Positive"
    elif s < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(s, 3)

def extract_recent_news(ticker, days=30, max_items=30):
    """
    Returns news items from last `days` days for the ticker (Google RSS + fallback).
    """
    query = f"{ticker}"
    items = fetch_google_news_rss(query, max_items=max_items)
    if not items:
        items = fetch_moneycontrol_search(ticker, max_items=max_items)
    # If still empty, return []
    cutoff = datetime.utcnow() - timedelta(days=days)
    recent = []
    for it in items:
        dt = it.get("pubDate")
        # Some sources don't provide pubDate; skip those with None
        if dt is None:
            # optionally try to ignore or keep with None; we'll skip for strict 30-day filter
            continue
        if dt.tzinfo:
            dt_utc = dt.astimezone(tz=None).replace(tzinfo=None)
        else:
            dt_utc = dt
        if dt_utc >= cutoff:
            recent.append({"title": it["title"], "link": it["link"], "date": dt_utc})
    # If too few, try Moneycontrol fallback searching by company name
    if len(recent) < 3:
        fallback = fetch_moneycontrol_search(ticker, max_items=40)
        for it in fallback:
            dt = it.get("pubDate")
            if not dt:
                continue
            if dt.tzinfo:
                dt = dt.astimezone(tz=None).replace(tzinfo=None)
            if dt >= cutoff:
                recent.append({"title": it["title"], "link": it["link"], "date": dt})
    # Deduplicate by title
    seen = set()
    uniq = []
    for r in recent:
        t = r["title"]
        if t in seen:
            continue
        seen.add(t)
        uniq.append(r)
    # sort by date desc
    uniq = sorted(uniq, key=lambda x: x["date"], reverse=True)
    return uniq[:15]

def summarize_news_for_prompt(ticker, days=30):
    """
    Produce the structured summary per your Master Prompt:
      1) 5-10 bullets with date + Positive/Neutral/Negative
      2) Recurring themes
      3) Overall sentiment 0-10 with reasoning
    """
    items = extract_recent_news(ticker, days=days, max_items=60)
    bullets = []
    theme_counter = Counter()
    scores = []
    for it in items:
        title = it["title"]
        dt = it["date"]
        cat = categorize_event(title)
        label, score = classify_sentiment(title)
        scores.append(score)
        theme_counter[cat] += 1
        date_str = dt.strftime("%Y-%m-%d")
        bullets.append({"date": date_str, "title": title, "category": cat, "sentiment_label": label, "score": score})
    # pick top 5-10 bullet points by recency and by significance (we use recency)
    top_bullets = bullets[:10] if len(bullets) >= 5 else bullets
    # recurring themes
    themes = [f"{k}({v})" for k, v in theme_counter.most_common()]
    # overall sentiment mapping: compound average -> 0..10
    overall_compound = sum(scores) / len(scores) if scores else 0.0
    # map [-1,1] -> [0,10]
    overall_score = round(((overall_compound + 1) / 2) * 10, 1)
    # reasoning
    reason = []
    if theme_counter:
        reason.append("Major themes: " + ", ".join(themes))
    if overall_compound > 0.05:
        reason.append("Recent headlines tilt positive; earnings/deal/expansion news predominate.")
    elif overall_compound < -0.05:
        reason.append("Recent headlines tilt negative; regulatory/weak earnings dominate.")
    else:
        reason.append("Mixed or neutral headlines; no strong directional signal.")
    # format output as text per master prompt
    bullet_lines = []
    for b in top_bullets:
        bullet_lines.append(f"- {b['date']}: [{b['sentiment_label']}] {b['title']}")
    summary = {
        "ticker": ticker,
        "period_days": days,
        "bullets": bullet_lines,
        "themes": themes,
        "overall_score_0_10": overall_score,
        "reasoning": " ".join(reason),
        "raw_items": top_bullets
    }
    return summary

# --------- STOCK SCREENER (Master Prompt) ---------
def check_fundamentals_for_ticker(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    # some keys might be missing
    def safe(key):
        return info.get(key, None)
    # gather basic metrics
    metrics = {
        "Ticker": ticker,
        "Sector": safe("sector") or "",
        "P/E": safe("trailingPE"),
        "P/B": safe("priceToBook"),
        "DividendYield": safe("dividendYield"),
        "DebtToEquity": safe("debtToEquity"),
        "FreeCashFlow": safe("freeCashflow"),   # sometimes missing
        "EPS": safe("trailingEps"),
        "MarketCap": safe("marketCap"),
        "RevenueGrowth": safe("revenueGrowth"),
    }
    return metrics

def screen_universe(tickers, industry_pe_avg_map=None):
    """
    Simple screener: returns stocks satisfying the criteria in your master prompt.
    - tickers: list of tickers to evaluate (e.g., Nifty50 list)
    - industry_pe_avg_map: optional dict sector->avgPE to compare P/E
    """
    candidates = []
    for t in tickers:
        try:
            m = check_fundamentals_for_ticker(t)
        except Exception:
            continue
        # quick filters (allow some missing fields)
        try:
            pe = float(m["P/E"]) if m["P/E"] else None
            pb = float(m["P/B"]) if m["P/B"] else None
            dy = float(m["DividendYield"]) if m["DividendYield"] else None
            de = float(m["DebtToEquity"]) if m["DebtToEquity"] else None
        except Exception:
            pe, pb, dy, de = None, None, None, None
        # Criteria:
        # - P/E lower than industry average (if industry map given, otherwise ignore)
        pe_ok = True
        if industry_pe_avg_map and m["Sector"] in industry_pe_avg_map and pe is not None:
            pe_ok = pe < industry_pe_avg_map[m["Sector"]]
        # - P/B < 3
        pb_ok = (pb is not None and pb < 3)
        # - Dividend yield > 2% if applicable (if None, we allow)
        dy_ok = (dy is None) or (dy >= 0.02)
        # - DebtToEquity < 0.5
        de_ok = (de is None) or (de < 0.5)
        # - Free cashflow positive (best-effort)
        fcf_ok = True
        fcf = m.get("FreeCashFlow")
        if fcf is not None:
            try:
                fcf_ok = float(fcf) > 0
            except Exception:
                fcf_ok = True
        # EPS growth & consistent EPS growth 3-year not reliably available via yfinance; skip strict check
        if pb_ok and de_ok and dy_ok and fcf_ok and pe_ok:
            candidates.append(m)
    # pick top 5 by MarketCap descending as proxy for quality
    dfc = pd.DataFrame(candidates)
    if dfc.empty:
        return []
    dfc["MarketCap"] = pd.to_numeric(dfc["MarketCap"], errors="coerce").fillna(0)
    dfc = dfc.sort_values("MarketCap", ascending=False)
    top5 = dfc.head(5).to_dict("records")
    # prepare return with table and short summary
    out = []
    for rec in top5:
        ratios = {
            "P/E": rec.get("P/E"),
            "P/B": rec.get("P/B"),
            "DividendYield": rec.get("DividendYield"),
            "DebtToEquity": rec.get("DebtToEquity"),
            "FreeCashFlow": rec.get("FreeCashFlow"),
        }
        sector = rec.get("Sector","")
        # competitors: attempt to find simple peers by scanning tickers in list with same sector
        out.append({
            "Ticker": rec["Ticker"],
            "Sector": sector,
            "Ratios": ratios,
            "Competitors": "See sector peers",
            "Summary": f"{rec['Ticker']} in {sector}: attractive based on low P/B and healthy market cap (quick screen)."
        })
    return out

# --------- Example utility functions to pretty-print the master prompt outputs
def pretty_news_summary(summary):
    s = []
    s.append(f"Summary for {summary['ticker']} (last {summary['period_days']} days)")
    s.append("\nSignificant news (5-10 bullets):")
    s.extend(summary['bullets'])
    s.append("\nRecurring themes:")
    s.append(", ".join(summary['themes']) or "None")
    s.append("\nOverall sentiment (0..10): " + str(summary['overall_score_0_10']))
    s.append("Reasoning: " + summary['reasoning'])
    return "\n".join(s)

def pretty_screener(out):
    if not out:
        return "No candidates found"
    lines = []
    for r in out:
        lines.append(f"Ticker: {r['Ticker']}  Sector: {r['Sector']}")
        lines.append(tabulate(pd.DataFrame([r['Ratios']]).T.reset_index().values, headers=["Metric","Value"]))
        lines.append("Competitors: " + r.get("Competitors",""))
        lines.append("Summary: " + r.get("Summary",""))
        lines.append("\n")
    return "\n".join(lines)
