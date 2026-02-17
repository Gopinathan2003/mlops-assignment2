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


# --------------------------------
# 2. Text-to-Image Endpoint (fal.ai example)
# --------------------------------
FAL_API_KEY = os.getenv("FAL_API_KEY")

class ImageRequest(BaseModel):
    prompt: str
    model: str = 'fal-ai/flux/dev'
    width: int = 1024
    height: int = 1024

@app.post("/generate-image")
async def generate_image(req: ImageRequest):
    if not FAL_API_KEY:
        raise HTTPException(500, "fal.ai API key not configured")

    payload = {
        "prompt": req.prompt,
        "image_size": f"{req.width}x{req.height}",
        "model": req.model,
        "api_key": FAL_API_KEY,
    }

    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        # fal.ai uses queue + webhook, but for simple sync we use /fal/v1/calls
        r = requests.post(
            f"https://queue.fal.run/{req.model}",
            json = payload,
            headers=headers,
        )
        r.raise_for_status()
        status_url = r.json()["status"]

        # Poll status (simplified; in production use websocket or retry)
        import time
        for _ in range(30):
            s = requests.get(status_url, headers=headers).json()
            if s["status"] == "COMPLETED":
                image_url = s["images"][0]["url"]
                return {"prompt": req.prompt, "image_url": image_url}
            if s["status"] in ["FAILED", "CANCELLED"]:
                raise Exception("Generation failed")
            time.sleep(2)
        raise Exception("Timeout waiting for image")
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)