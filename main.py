from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from googletrans import Translator
import requests
import os
import inspect
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

app = FastAPI(title="AI Microservices - FastAPI Demo", version="0.1.0")

async def _resolve_awaitable(value, max_depth: int = 5):
    for _ in range(max_depth):
        if inspect.isawaitable(value):
            value = await value
        else:
            break
    return value

# -------------------------------
# 1. Translation Endpoint
# -------------------------------
class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "fr"  # e.g. "fr", "es", "de", "zh"

# DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
# DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")  # Get free key → https://www.deepl.com/pro-api?cta=header-pro-api



@app.post("/translate")
async def translate_text(req: TranslateRequest):
    try:
        async with Translator() as translator:
            result = await _resolve_awaitable(
                translator.translate(req.text, dest=req.target_lang.lower())
            )
            translated = await _resolve_awaitable(getattr(result, "text", result))

        return {"original": req.text, "translated": translated, "target_lang": req.target_lang}
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")


# --------------------------------
# 2. Text-to-Image Endpoint (fal.ai example)
# --------------------------------

class ImageRequest(BaseModel):
    prompt: str
    # Optional Stability engine_id; if omitted, server will use the first available engine.
    model: str | None = None
    width: int = 512
    height: int = 512

@app.post("/generate-image")
async def generate_image(req: ImageRequest):
    sta_api_key = os.getenv("STA_API_KEY")
    if not sta_api_key:
        raise HTTPException(500, "API key not configured. Set STA_API_KEY and restart the server.")

    try:
        engines_resp = requests.get(
            "https://api.stability.ai/v1/engines/list",
            headers={"Authorization": f"Bearer {sta_api_key}", "Accept": "application/json"},
            timeout=20,
        )
        engines_resp.raise_for_status()
        engines_data = engines_resp.json()
        engine_ids = [e.get("id") for e in engines_data if e.get("id")]

        if not engine_ids:
            raise HTTPException(500, "No Stability engines available for this API key.")

        model = req.model or engine_ids[0]
        if model not in engine_ids:
            raise HTTPException(
                400,
                f"Unknown model '{model}'. Use one of: {', '.join(engine_ids)}",
            )

        # Call Stability AI text-to-image endpoint
        url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
        headers = {
            "Authorization": f"Bearer {sta_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "text_prompts": [{"text": req.prompt, "weight": 1}],
            "height": req.height,
            "width": req.width,
            "cfg_scale": 7,
            "steps": 30,
            "samples": 1,
        }

        r = requests.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Generation failed: {r.status_code} {r.text}")

        data = r.json()
        if "artifacts" not in data or not data["artifacts"]:
            raise HTTPException(500, f"Generation failed: malformed response {data}")

        import base64
        image_base64 = data["artifacts"][0].get("base64")
        if not image_base64:
            raise HTTPException(500, "Generation failed: missing image data")

        image_bytes = base64.b64decode(image_base64)

        # Save returned image bytes
        images_dir = Path("generated_images")
        images_dir.mkdir(parents=True, exist_ok=True)

        import time as _time
        timestamp = int(_time.time())
        filename = images_dir / f"image_{timestamp}.png"

        with open(filename, "wb") as f:
            f.write(image_bytes)

        return {
            "prompt": req.prompt,
            "model": model,
            "saved_path": str(filename),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
