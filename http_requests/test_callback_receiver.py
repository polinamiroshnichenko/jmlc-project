"""
Local webhook receiver for manual testing of the async checkup-assistant API.

Run:
    uvicorn test_callback_receiver:app --port 9001

Then POST to http://localhost:8000/cds-services/checkup-assistant/ with:
    "callback_url": "http://localhost:9001/callback"
"""

import json
import logging

from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/callback")
async def receive_callback(request: Request):
    body = await request.json()
    logger.info(json.dumps(body, ensure_ascii=False, indent=2))
    return {"ok": True}
