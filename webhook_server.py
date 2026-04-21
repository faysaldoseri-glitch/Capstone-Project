"""
webhook_server.py – FastAPI endpoint for receiving SMS via webhook.

Run:  uvicorn webhook_server:api --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request

from parser import parse_sms_block
from db     import init_db, save_transactions

init_db()
api = FastAPI(title="Budget App Webhook", version="2.0")


@api.post("/webhook/sms")
async def receive_sms(request: Request):
    try:
        payload  = await request.json()
        sms_text = payload.get("text", "")

        if not sms_text:
            return {"status": "ignored", "reason": "Empty text"}

        df = parse_sms_block(sms_text)

        if df.empty:
            return {"status": "ignored", "reason": "No parsable transactions"}

        n = save_transactions(df)
        return {"status": "success", "saved": n}

    except Exception as e:
        return {"status": "error", "message": str(e)}
