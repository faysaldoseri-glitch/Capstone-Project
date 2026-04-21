"""
services.py – External service integrations for the Bahrain Budget App.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from huggingface_hub import InferenceClient

from config import BOT_TOKEN, CHAT_ID, HF_API_KEY, ALERT_STATE_FILE


# ── Telegram ─────────────────────────────────────────────────────────
def send_telegram(message: str, bot_token: str = BOT_TOKEN, chat_id: str = CHAT_ID) -> tuple[int, str]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=20)
    return r.status_code, r.text


def format_summary(df: pd.DataFrame, budget: float, month: str) -> str:
    if df.empty:
        return "No transactions parsed yet."
    dfm = df.copy()
    dfm["month"] = pd.to_datetime(dfm["date"]).dt.to_period("M").astype(str)
    dfm = dfm[dfm["month"] == month]
    if dfm.empty:
        return f"No transactions found for {month}."

    total = float(dfm["amount_bhd"].sum())
    remaining = budget - total
    status = "On track" if remaining >= 0 else "Over budget"
    top_cat = dfm.groupby("category")["amount_bhd"].sum().sort_values(ascending=False)
    top_merch = dfm.groupby("merchant")["amount_bhd"].sum().sort_values(ascending=False)

    return (
        f"Budget Summary ({month})\n"
        f"Spent: {total:.3f} BHD\n"
        f"Budget: {budget:.3f} BHD\n"
        f"Remaining: {remaining:.3f} BHD\n"
        f"Status: {status}\n\n"
        f"Top category: {top_cat.index[0]} ({float(top_cat.iloc[0]):.3f} BHD)\n"
        f"Top merchant: {top_merch.index[0]} ({float(top_merch.iloc[0]):.3f} BHD)"
    )


# ── Auto budget alerts ───────────────────────────────────────────────
def _load_alert_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_alert_state(state: dict) -> None:
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def maybe_send_budget_alert(df: pd.DataFrame, budget: float, month: str) -> str | None:
    if df.empty or budget <= 0:
        return None
    if not BOT_TOKEN or BOT_TOKEN.startswith("PASTE_"):
        return None
    if not CHAT_ID or CHAT_ID.startswith("PASTE_"):
        return None

    dfm = df.copy()
    dfm["month"] = pd.to_datetime(dfm["date"]).dt.to_period("M").astype(str)
    dfm = dfm[dfm["month"] == month]
    if dfm.empty:
        return None

    total = float(dfm["amount_bhd"].sum())
    ratio = total / budget
    today = datetime.today()

    alert_key, message = None, None

    if today.day <= 10 and ratio >= 0.70:
        alert_key = f"{month}_early_70"
        message = f"Early Warning ({month})\nAlready spent {total:.3f}/{budget:.3f} BHD in the first 10 days!"
    elif ratio >= 1.00:
        alert_key = f"{month}_over_100"
        message = f"Over Budget ({month})\nSpent: {total:.3f} BHD | Budget: {budget:.3f} BHD"
    elif ratio >= 0.90:
        alert_key = f"{month}_warn_90"
        message = f"Nearing Limit ({month})\nSpent {total:.3f}/{budget:.3f} BHD"

    if not alert_key:
        return None

    state = _load_alert_state()
    if state.get(alert_key):
        return None

    code, _ = send_telegram(message)
    if code == 200:
        state[alert_key] = datetime.now().isoformat()
        _save_alert_state(state)
        return "Automatic Telegram alert sent."
    return f"Alert failed (HTTP {code})."


# ── OCR (PaddleOCR — supports Arabic + English, runs locally) ────────
_ocr_engine = None


def _get_ocr_engine():
    """Lazy-load PaddleOCR so the app starts fast."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    from paddleocr import PaddleOCR
    _ocr_engine = PaddleOCR(lang="en")
    return _ocr_engine


def ocr_image(image_bytes: bytes, **_kwargs) -> str:
    """
    Extract text from a receipt image using PaddleOCR.
    Runs locally — no API key needed, works offline.
    Supports Arabic and English.
    """
    import tempfile, os

    # PaddleOCR needs a file path, so write bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr_engine()
        result = ocr.ocr(tmp_path)

        # Extract text lines from PaddleOCR result
        lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0] if line[1] else ""
                if text.strip():
                    lines.append(text.strip())

        return "\n".join(lines)
    finally:
        os.unlink(tmp_path)


# ── Voice transcription ─────────────────────────────────────────────
# Model priority list — try each until one works
_WHISPER_MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-small",
]


def transcribe_audio(audio_path: str, language: str = "en", api_key: str = HF_API_KEY) -> str:
    """
    Transcribe audio using HuggingFace Whisper.
    language: 'en' for English, 'ar' for Arabic
    Tries multiple models in case one is unavailable.
    """
    if not api_key or api_key.startswith("PASTE_"):
        raise RuntimeError("No valid Hugging Face API key configured.")

    client = InferenceClient(token=api_key)

    last_error = None
    for model_id in _WHISPER_MODELS:
        try:
            result = client.automatic_speech_recognition(
                audio_path,
                model=model_id,
            )
            return result.text
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All Whisper models failed. Last error: {last_error}")