# #Combine NIFTY + SENSEX + Indian economy keywords
# market_keywords = "NIFTY 50 OR SENSEX OR Indian economy OR RBI OR inflation OR stock market India"

# # --- 1️⃣ Fetch AI-based News Sentiment (last 3 days) ---
# summary = summarize_news_for_prompt(market_keywords, days=3)

# if summary and "bullets" in summary:
#     st.markdown("### 📰 Market News Summary (Last 3 Days)")
#     for b in summary["bullets"]:
#         st.write(b)

#     sentiment_score = summary.get("overall_score_0_10", 5)
#     st.markdown(f"**🧩 Themes:** {', '.join(summary.get('themes', []))}")
#     st.markdown(f"**🧠 Market Sentiment Score (0–10):** {sentiment_score:.1f}")
# else:
#     st.warning("⚠️ Could not fetch market news sentiment.")
#     sentiment_score = 5

# # --- 2️⃣ Fetch Live Market Performance (NIFTY & SENSEX) ---
# try:
#     nifty = yf.Ticker("^NSEI").history(period="2d")["Close"]
#     sensex = yf.Ticker("^BSESN").history(period="2d")["Close"]

#     if len(nifty) >= 2 and len(sensex) >= 2:
#         nifty_change = (nifty.iloc[-1] - nifty.iloc[-2]) / nifty.iloc[-2] * 100
#         sensex_change = (sensex.iloc[-1] - sensex.iloc[-2]) / sensex.iloc[-2] * 100
#         avg_change = (nifty_change + sensex_change) / 2
#     else:
#         avg_change = 0
# except Exception as e:
#     st.warning("⚠️ Could not fetch live market data.")
#     avg_change = 0

# # --- 3️⃣ Combine Both Signals ---
# # Blend news sentiment (70%) + live market (30%)
# # Convert avg_change (approx % move) into a 0–10 scale
#     normalized_market = 5 + avg_change  # 0% = 5, +5% = 10, -5% = 0
#     final_score = 0.7 * sentiment_score + 0.3 * max(0, min(10, normalized_market))

# # --- 4️⃣ Interpret Combined Signal ---
#     if final_score >= 6:
#         trend = "📈 Likely Uptrend"
#         color = "green"
#     elif final_score <= 4:
#         trend = "📉 Likely Downtrend"
#         color = "red"
#     else:
#         trend = "⚖️ Neutral / Sideways"
#         color = "gray"

# # --- 5️⃣ Display Combined Market View ---
#     st.markdown("### 🧭 Combined Market Direction Indicator")
#     st.markdown(
#         f"**Market Mood (Last 3 Days):** {sentiment_score:.1f}/10  \n"
#         f"**NIFTY Change:** {nifty_change:.2f}%  |  **SENSEX Change:** {sensex_change:.2f}%")
#     st.markdown(
#         f"### **Predicted Market Direction:** <span style='color:{color}'>{trend}</span>",unsafe_allow_html=True,)
#     st.caption("➡️ Based on blended sentiment + live market movement. Updates daily.")