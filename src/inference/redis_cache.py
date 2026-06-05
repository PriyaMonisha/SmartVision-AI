# filename: src/inference/redis_cache.py
# purpose:  Redis cache-aside pattern for classify and detect endpoints.
#           Rule 23: never raise on Redis errors — inference continues without cache.
#           Rule 31 (socket timeout): 1s connect timeout prevents blocking event loop
#           when Redis is not running locally.

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RedisCache:
    """SHA256 cache-aside with graceful degradation on Redis unavailability.

    Cache hit: returns dict (caller must pop 'cached' key before unpacking into response).
    Cache miss or error: returns None — inference runs normally.
    """

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        password: str = "",
        timeout: float = 1.0,
        *,
        _client=None,
    ) -> None:
        self._available = False
        self._client = None
        try:
            if _client is not None:
                # Injected client (used in tests via fakeredis)
                self._client = _client
                self._client.ping()
            else:
                import redis as redis_lib

                self._client = redis_lib.Redis(
                    host=host,
                    port=port,
                    password=password
                    or None,  # None = no auth (empty string means no auth)
                    socket_connect_timeout=timeout,  # fail fast — not 20-30s OS timeout
                    socket_timeout=timeout,
                    decode_responses=False,  # raw bytes for json.loads
                )
                self._client.ping()
            self._available = True
            logger.info(f"Redis connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({host}:{port}): {e}. Cache disabled.")

    @property
    def available(self) -> bool:
        return self._available

    def get(self, key: str) -> Optional[dict]:
        """Return cached dict or None on miss / connection error. Never raises."""
        if not self._available:
            return None
        try:
            val = self._client.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning(f"Redis get error (key={key[:20]}...): {e}")
            return None

    def set(self, key: str, value: dict, ttl: int) -> None:
        """Store value with TTL. Silent on any Redis error. Never raises."""
        if not self._available:
            return
        try:
            self._client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Redis set error (key={key[:20]}...): {e}")

    def push_to_list(self, key: str, value: float, max_len: int) -> None:
        """LPUSH + LTRIM in pipeline (single round-trip).

        NOT atomic (no MULTI/EXEC). Safe for single-worker uvicorn deployments.
        With multiple workers the list may transiently exceed max_len between
        the two commands — the caller's in-process deque provides the safety net.
        """
        if not self._available:
            return
        try:
            pipe = self._client.pipeline()
            pipe.lpush(key, str(value))
            pipe.ltrim(key, 0, max_len - 1)
            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis push_to_list error (key={key}): {e}")

    def get_list(self, key: str) -> list[float]:
        """LRANGE 0 -1 → decode bytes → list[float].

        Returns [] on any error or missing key.
        decode_responses=False: items are bytes, decoded before float().
        Items are in LPUSH (newest-first) order — caller must reverse()
        before extending a deque to preserve chronological insertion order.
        """
        if not self._available:
            return []
        try:
            raw: list[bytes] = self._client.lrange(key, 0, -1)
            return [float(v.decode("utf-8")) for v in raw]
        except Exception as e:
            logger.warning(f"Redis get_list error (key={key}): {e}")
            return []

    @staticmethod
    def make_classify_key(
        image_bytes: bytes, model_name: str, model_hash: str = ""
    ) -> str:
        """32-char SHA256 prefix (128-bit) — negligible birthday collision probability."""
        img = hashlib.sha256(image_bytes).hexdigest()[:32]
        return f"sv:classify:{img}:{model_name}:{model_hash}"

    @staticmethod
    def make_detect_key(image_bytes: bytes, conf: float) -> str:
        img = hashlib.sha256(image_bytes).hexdigest()[:32]
        return f"sv:detect:{img}:{conf:.2f}"
