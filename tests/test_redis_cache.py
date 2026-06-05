"""
tests/test_redis_cache.py
─────────────────────────
Unit tests for RedisCache (10 tests).

Uses fakeredis for real Redis semantics without a running server.
RedisCache._client is injected via the _client= keyword argument added in
Section 12 (src/inference/redis_cache.py).

Key implementation facts
------------------------
- make_classify_key(image_bytes, model_name, model_hash="")
    → "sv:classify:<sha256[:32]>:<model_name>:<model_hash>"
- make_detect_key(image_bytes, conf)
    → "sv:detect:<sha256[:32]>:<conf:.2f>"
- push_to_list stores str(float_value) via LPUSH; get_list returns list[float]
  in LPUSH order (newest first)
- LTRIM(0, max_len - 1) keeps the max_len most-recently-pushed items
"""

from __future__ import annotations

from unittest.mock import MagicMock


from src.inference.redis_cache import RedisCache


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_cache(fake_redis) -> RedisCache:
    """Build a RedisCache backed by fakeredis."""
    return RedisCache(_client=fake_redis)


# ── get / set round-trips ──────────────────────────────────────────────────────


def test_get_returns_none_on_cache_miss(fake_redis_client) -> None:
    cache = _make_cache(fake_redis_client)
    assert cache.get("nonexistent_key") is None


def test_set_then_get_returns_original_dict(fake_redis_client) -> None:
    cache = _make_cache(fake_redis_client)
    data = {"predictions": [{"class_name": "cat", "confidence": 0.95}]}
    cache.set("my_key", data, ttl=60)
    assert cache.get("my_key") == data


def test_set_get_survives_complex_dict(fake_redis_client) -> None:
    """Nested dicts with lists round-trip through JSON serialization intact."""
    cache = _make_cache(fake_redis_client)
    data = {
        "detections": [
            {
                "class_name": "dog",
                "confidence": 0.92,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 50.0,
                "y2": 80.0,
            }
        ],
        "inference_time_ms": 38.2,
    }
    cache.set("detect_key", data, ttl=120)
    retrieved = cache.get("detect_key")
    assert retrieved == data


# ── Graceful degradation ───────────────────────────────────────────────────────


def test_redis_unavailable_available_is_false() -> None:
    """Connection error during ping → _available=False."""
    bad_client = MagicMock()
    bad_client.ping.side_effect = Exception("Connection refused")
    cache = RedisCache(_client=bad_client)
    assert cache._available is False


def test_redis_unavailable_get_returns_none() -> None:
    """When unavailable, get() returns None without raising."""
    bad_client = MagicMock()
    bad_client.ping.side_effect = Exception("Connection refused")
    cache = RedisCache(_client=bad_client)
    assert cache.get("any_key") is None  # must not raise


def test_redis_unavailable_set_is_noop() -> None:
    """When unavailable, set() completes silently without raising."""
    bad_client = MagicMock()
    bad_client.ping.side_effect = Exception("Connection refused")
    cache = RedisCache(_client=bad_client)
    cache.set("any_key", {"foo": "bar"}, ttl=60)  # must not raise


# ── Key format helpers ─────────────────────────────────────────────────────────


def test_make_classify_key_format(fake_redis_client) -> None:
    """sv:classify:<32-hex>:<model_name>:<model_hash>"""
    key = RedisCache.make_classify_key(b"test image bytes", "resnet50", "abcd1234")
    parts = key.split(":")
    assert parts[0] == "sv"
    assert parts[1] == "classify"
    assert len(parts[2]) == 32, "SHA256 prefix must be 32 hex chars"
    assert parts[3] == "resnet50"
    assert parts[4] == "abcd1234"


def test_make_detect_key_format(fake_redis_client) -> None:
    """sv:detect:<32-hex>:<conf:.2f>"""
    key = RedisCache.make_detect_key(b"test image bytes", 0.75)
    parts = key.split(":")
    assert parts[0] == "sv"
    assert parts[1] == "detect"
    assert len(parts[2]) == 32
    assert parts[3] == "0.75"  # exactly 2 decimal places


# ── List operations (drift monitoring) ────────────────────────────────────────


def test_push_to_list_and_get_list_round_trip(fake_redis_client) -> None:
    """push_to_list (float) → get_list returns list[float]."""
    cache = _make_cache(fake_redis_client)
    cache.push_to_list("drift:person", 0.85, max_len=100)
    result = cache.get_list("drift:person")
    assert len(result) == 1
    assert abs(result[0] - 0.85) < 1e-6


def test_push_to_list_enforces_maxlen(fake_redis_client) -> None:
    """After pushing 15 items with maxlen=10, list has exactly 10 entries (newest)."""
    cache = _make_cache(fake_redis_client)
    for i in range(15):
        cache.push_to_list("test_list", float(i), max_len=10)
    result = cache.get_list("test_list")
    assert len(result) == 10
    # LPUSH adds to front → newest item (14) is first
    assert abs(result[0] - 14.0) < 1e-6
    assert abs(result[-1] - 5.0) < 1e-6
