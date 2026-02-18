"""
Simple webhook endpoint for WhatsApp messages.
Receives messages and pushes to Redis queue for worker processing.

Security:
- Webhook signature validation (HMAC-SHA256) when VERIFY_WHATSAPP_SIGNATURE=True
- Verify token from environment variable WHATSAPP_VERIFY_TOKEN
"""

import hmac
import hashlib
from fastapi import APIRouter, Request, HTTPException
import redis.asyncio as aioredis
import logging

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify Infobip webhook signature using HMAC-SHA256.

    Args:
        payload: Raw request body bytes
        signature: Signature from X-Hub-Signature-256 header
        secret: INFOBIP_SECRET_KEY from environment

    Returns:
        True if signature is valid, False otherwise
    """
    if not secret:
        logger.warning("INFOBIP_SECRET_KEY not configured, skipping signature validation")
        return True

    if not signature:
        logger.warning("No signature header in webhook request")
        return False

    # Infobip uses sha256=<hex_digest> format
    if signature.startswith("sha256="):
        signature = signature[7:]

    expected = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected.lower(), signature.lower())

router = APIRouter()

# Async Redis client (lazy initialization)
_redis_client = None


async def get_redis():
    """Get async Redis client (lazy initialization)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    return _redis_client


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Receive WhatsApp webhook messages and push to Redis STREAM.

    Security:
    - Validates HMAC-SHA256 signature when VERIFY_WHATSAPP_SIGNATURE=True
    - Signature header: X-Hub-Signature-256

    Flow:
    1. Validate webhook signature (if enabled)
    2. Receive webhook from WhatsApp
    3. Extract message data (sender, text, message_id)
    4. VALIDATE sender is present (prevents 400 errors in WhatsApp response)
    5. Push to Redis STREAM: "whatsapp_stream_inbound"
    6. Worker picks up from stream via consumer group
    """
    try:
        # Get raw body for signature validation
        raw_body = await request.body()

        # Validate signature if enabled
        if settings.VERIFY_WHATSAPP_SIGNATURE:
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not verify_webhook_signature(raw_body, signature, settings.INFOBIP_SECRET_KEY):
                logger.warning(
                    f"Invalid webhook signature from {request.client.host}"
                )
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse JSON body
        import json
        body = json.loads(raw_body)

        logger.info(f"Received WhatsApp webhook: {body}")

        # Extract message details from Infobip format
        results = body.get("results", [])
        if not results:
            logger.warning("No results in webhook body")
            return {"status": "ok"}

        for result in results:
            # Infobip uses "from" for sender phone number
            # Fallback to "sender" for older API versions
            sender = result.get("from", "") or result.get("sender", "")
            message_id = result.get("messageId", "")

            # Skip delivery/seen reports (no "from" field)
            if not sender:
                logger.info(
                    f"No sender in webhook result (delivery/seen report?) "
                    f"message_id={message_id}, keys={list(result.keys())}"
                )
                continue

            # Extract text - try all known Infobip formats
            text = ""

            # Format 1: "message" object (standard Infobip WhatsApp incoming)
            # {"message": {"type": "TEXT", "text": "Hello"}}
            msg = result.get("message")
            if isinstance(msg, dict):
                text = msg.get("text", "")

            # Format 2: "content" as list of objects
            # {"content": [{"type": "TEXT", "text": "Hello"}]}
            if not text:
                content = result.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "TEXT":
                            text = item.get("text", "")
                            break
                elif isinstance(content, dict):
                    # "content" as single object: {"content": {"type": "TEXT", "text": "Hello"}}
                    if content.get("type") == "TEXT":
                        text = content.get("text", "")

            # Format 3: direct "text" field (some Infobip integrations)
            if not text:
                text = result.get("text", "")

            if not text:
                msg_type = msg.get("type", "UNKNOWN") if isinstance(msg, dict) else "N/A"
                logger.warning(
                    f"No text in message from {sender[-4:]}... "
                    f"type={msg_type}, keys={list(result.keys())}"
                )
                continue

            # Push to Redis STREAM (not list!) - this is what worker listens to
            stream_data = {
                "sender": sender,
                "text": text,
                "message_id": message_id
            }

            redis = await get_redis()
            await redis.xadd("whatsapp_stream_inbound", stream_data)

            logger.info(f"Message pushed to stream: {sender[-4:]}... - {text[:30]}")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/whatsapp")
async def whatsapp_webhook_verify(request: Request):
    """
    WhatsApp webhook verification endpoint.

    Infobip sends GET to verify the webhook URL is reachable.
    If WHATSAPP_VERIFY_TOKEN is configured, validates hub.verify_token.
    Otherwise, returns 200 OK to pass Infobip's reachability check.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    expected_token = settings.WHATSAPP_VERIFY_TOKEN

    # No verify token configured - just confirm the URL is reachable
    if not expected_token:
        logger.info("Webhook verification: no token configured, returning OK")
        if challenge:
            return int(challenge)
        return {"status": "ok", "webhook": "active"}

    # Token configured - validate it
    if mode == "subscribe" and token == expected_token:
        logger.info("WhatsApp webhook verified successfully")
        return int(challenge)

    logger.warning(
        f"WhatsApp webhook verification failed: mode={mode}, "
        f"token_match={token == expected_token if expected_token else 'no_token'}"
    )
    raise HTTPException(status_code=403, detail="Verification failed")
