from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os


app = FastAPI(title="AI Microservices - FastAPI Demo", version="0.1.0")

# -------------------------------
# 1. Translation Endpoint
# -------------------------------
class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "fr"  # e.g. "fr", "es", "de", "zh"

DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")  # Get free key → https://www.deepl.com/pro-api?cta=header-pro-api

@app.post("/translate")
async def translate_text(req: TranslateRequest):
    if not DEEPL_API_KEY:
        raise HTTPException(500, "DeepL API key not configured")

    params = {
        "auth_key": DEEPL_API_KEY,
        "text": req.text,
        "target_lang": req.target_lang.upper(),
    }
    try:
        r = requests.post(DEEPL_FREE_URL, data=params)
        r.raise_for_status()
        result = r.json()
        translated = result["translations"][0]["text"]
        return {"original": req.text, "translated": translated, "target_lang": req.target_lang}
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)