import asyncio
import json
import os
import re
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from typing import List
from data_models.schemas import PeerInfo
from utils.ticker_utils import ticker_exists

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

async def _gemini_text(prompt: str) -> str:
    def _call():
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return (response.text or "").strip()

    return await asyncio.to_thread(_call)

def _extract_json_block(content: str):
    if "{" in content and "}" in content:
        content = content[content.find("{"):content.rfind("}") + 1]
    return json.loads(content)

async def get_nse_ticker_from_name(company_name):
    print(f"🔍 Looking up ticker for: {company_name}")
    try:
        prompt = f"""Identify the most accurate India National Stock Exchange (NSE) ticker symbol for the company: "{company_name}". 
        Return ONLY the ticker symbol followed by '.NS' (e.g., RELIANCE.NS, TCS.NS). 
        If it's not a prominent NSE stock or unknown, return 'NOT_FOUND'."""
        
        response_text = await _gemini_text(prompt)
        ticker_matches = re.findall(r"[A-Z0-9&-]+\.NS", response_text.upper())
        ticker = ticker_matches[0] if ticker_matches else response_text.strip().upper()
        
        if ticker != "NOT_FOUND" and ticker.endswith(".NS"):
            if ticker_exists(ticker):
                print(f"✅ AI returned valid symbol: {ticker}")
                return ticker
            else:
                print(f"⚠️ AI returned {ticker} but it doesn't exist on Yahoo Finance.")
        
        return "NOT_FOUND"
    except Exception as e:
        print(f"Ticker lookup error: {e}")
        return "NOT_FOUND"

async def get_sector_peers(ticker: str) -> List[str]:
    """Identifies top 3 NSE competitors using AI."""
    try:
        system_prompt = f"Identify the top 3 direct NSE-listed competitors for the stock {ticker}. Return ONLY the ticker symbols separated by commas (e.g., TCS.NS,INFY.NS,HCLTECH.NS). All tickers MUST end in .NS."
        response_text = await _gemini_text(system_prompt)
        peer_list = list(dict.fromkeys(re.findall(r"[A-Z0-9&-]+\.NS", response_text.upper())))
        print(f"👥 Identified peers: {peer_list}")
        return peer_list
    except Exception as e:
        print(f"Peer identification error: {e}")
        return []

async def get_gemini_comprehensive_analysis(ticker, latest_row, indicators, fundamentals, news, lstm_trend):
    """
    Asks Gemini to analyze the stock based on combined technical and news data.
    """
    news_titles = [item.title for item in news]
    
    prompt = f"""You are a professional stock analyst. Perform a deep-dive analysis for {ticker}.
    Current Price: {latest_row['Close']}
    Indicators: RSI={indicators.rsi:.2f}, SMA50={indicators.sma50:.2f}, EMA20={indicators.ema20:.2f}
    LSTM Prediction: {lstm_trend}
    Recent News: {news_titles}
    Sector: {fundamentals['sector']}
    P/E: {fundamentals['pe']}

    Return JSON strictly in this format:
    {{
      "summary": "2-3 sentences max on overall outlook",
      "sentiment_score": value from -1.0 to 1.0,
      "recommendation": "BUY", "SELL", or "HOLD",
      "news_impact": ["Short impact description for news 1", ...]
    }}
    """
    
    try:
        content = await _gemini_text(prompt)
        return _extract_json_block(content)
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return {
            "summary": "Technical analysis complete. Neural trend is " + lstm_trend,
            "sentiment_score": 0.0,
            "recommendation": "HOLD",
            "news_impact": ["N/A"] * len(news)
        }

async def fetch_peer_data(ticker: str) -> PeerInfo:
    """Asynchronously fetches key metrics for a single peer."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        info = stock.info
        if data.empty: return None
        
        latest_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change_pct = ((latest_price - prev_price) / prev_price) * 100
        
        return PeerInfo(
            ticker=ticker,
            price=float(latest_price),
            pe=info.get('trailingPE'),
            change_pct=float(change_pct)
        )
    except Exception:
        return None
