from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os

# Add the parent directory to the path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.engine import perform_full_analysis
from backend.services.ai_service import get_nse_ticker_from_name
from backend.cache_manager import cache

app = FastAPI(title="StockPulse Advanced API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    company_name: str

@app.get("/")
async def read_root():
    return {"message": "StockPulse AI API is online on Vercel."}

@app.post("/analyze")
async def analyze(data: AnalysisRequest):
    if not data.company_name:
        return JSONResponse(status_code=400, content={"error": "Company name is required"})
    
    ticker = await get_nse_ticker_from_name(data.company_name)
    if ticker == "NOT_FOUND":
        return {"error": "Ticker not found for this company"}

    cached_result = cache.get(ticker)
    if cached_result and "peers" in cached_result:
        print(f"✅ Cache Hit for {ticker}")
        return cached_result
    elif cached_result:
        print(f"🔄 Stale cache found for {ticker}. Re-running analysis...")
    
    result = await perform_full_analysis(data.company_name)
    
    if "error" not in result:
        trend = result.get("lstm_trend", "")
        if "Unknown" not in trend:
            cache.set(ticker, result)
            print(f"💾 Results cached for {ticker}")
    
    return result
