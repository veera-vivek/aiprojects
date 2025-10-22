# src/reporting.py
import yfinance as yf
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta import momentum, trend, volatility


REPORT_DIR = "reports"

def get_fundamentals(ticker):
    t = yf.Ticker(ticker)
    info = t.info
    keys = ['longName','sector','industry','marketCap','trailingPE','priceToBook',
            'returnOnEquity','debtToEquity','dividendYield','trailingEps','revenueGrowth']
    f = {k: info.get(k,'NA') for k in keys}
    return f

def get_technical_summary(price_df):
    df = price_df.copy().dropna()
    df['rsi'] = momentum.rsi(df['Close'])
    df['macd'] = trend.macd_diff(df['Close'])
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    last = df.iloc[-1]
    tech = {
        'RSI': round(last['rsi'],2) if pd.notnull(last['rsi']) else None,
        'MACD': round(last['macd'],4) if pd.notnull(last['macd']) else None,
        'MA20_vs_Close': round(last['Close'] - last['ma20'],4) if pd.notnull(last['ma20']) else None,
        'MA50_vs_Close': round(last['Close'] - last['ma50'],4) if pd.notnull(last['ma50']) else None,
        'Volatility(14d)': round(volatility.average_true_range(df['High'],df['Low'],df['Close'],14).iloc[-1],4)
    }
    return tech

def get_news_sentiment(ticker, count=5):
    """
    Fetch recent news and compute sentiment for a stock.
    Prioritizes relevant company-related news via company name search.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import requests
    from bs4 import BeautifulSoup
    import yfinance as yf

    analyzer = SentimentIntensityAnalyzer()
    items = []

    # --- Get company name for better relevance ---
    try:
        t = yf.Ticker(ticker)
        company_name = t.info.get("longName", ticker.replace(".NS", ""))
    except Exception:
        company_name = ticker.replace(".NS", "")

    # --- Try Yahoo Finance first ---
    try:
        news = t.news if hasattr(t, "news") else []
        if news:
            for n in news[:count]:
                title = n.get('title', '').strip()
                if not title:
                    continue
                score = analyzer.polarity_scores(title)['compound']
                items.append({'title': title, 'sentiment': round(score, 3)})
    except Exception as e:
        print(f"⚠️ Yahoo news failed for {ticker}: {e}")

    # --- Google News fallback with better relevance ---
    if len(items) < count:
        try:
            queries = [
                f'"{company_name}" stock site:moneycontrol.com OR site:economictimes.indiatimes.com',
                f'"{company_name}" share news',
                f'{ticker} NSE stock news'
            ]
            seen_titles = set()

            for query in queries:
                if len(items) >= count:
                    break
                url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
                response = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
                soup = BeautifulSoup(response.content, "lxml-xml")

                for item in soup.find_all("item"):
                    if len(items) >= count:
                        break
                    title = item.title.text.strip() if item.title else ""
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # Filter out irrelevant headlines (Apple, Cricket, etc.)
                    if not any(word.lower() in title.lower() for word in company_name.split()[:2]):
                        continue

                    score = analyzer.polarity_scores(title)['compound']
                    items.append({'title': title, 'sentiment': round(score, 3)})

        except Exception as e:
            print(f"⚠️ Google News fallback failed for {ticker}: {e}")

    return items[:count]

def get_peers(tickers_to_check):
    rows = []
    for t in tickers_to_check:
        info = yf.Ticker(t).info
        rows.append({
            'Ticker': t,
            'Sector': info.get('sector',''),
            'PE': info.get('trailingPE'),
            'PB': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'DivYield': info.get('dividendYield')
        })
    df = pd.DataFrame(rows)
    return df

def generate_report(ticker, price_df, peer_pool=None):
    os.makedirs(REPORT_DIR, exist_ok=True)
    fund = get_fundamentals(ticker)
    tech = get_technical_summary(price_df)
    news = get_news_sentiment(ticker)
    peers = get_peers(peer_pool if peer_pool else [ticker])

    md = []
    md.append(f"# Market Analysis Report — {fund.get('longName',ticker)}")
    md.append(f"**Ticker:** {ticker}  ")
    md.append(f"**Sector:** {fund.get('sector','NA')}  **Industry:** {fund.get('industry','NA')}  ")
    md.append("## 🧭 Fundamental Overview")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    for k,v in fund.items():
        if k in ['longName','sector','industry']: continue
        md.append(f"| {k} | {v} |")
    md.append("\n## 📊 Technical Indicators")
    md.append("| Indicator | Value |")
    md.append("|---|---|")
    for k,v in tech.items():
        md.append(f"| {k} | {v} |")
    md.append("\n## 🗞 News & Sentiment")
    if news:
        for n in news:
           s = n['sentiment']
           label = "🟢 Positive" if s > 0.05 else "🔴 Negative" if s < -0.05 else "⚪ Neutral"
           md.append(f"- {label} ({s:+.2f}) {n['title']}")
    else:
        md.append("- No news items found")
    md.append("\n## 🧩 Peer Comparison")
    md.append(peers.to_markdown(index=False))
    report_path = os.path.join(REPORT_DIR, f"{ticker.replace('.','_')}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))
    return report_path
