import copy
import hashlib
import json
import threading
import time


DEFAULT_TTL_SECONDS = 60 * 30
_cache = {}
_locks = {}
_locks_guard = threading.Lock()


def normalize_text(text):
    return " ".join(str(text or "").split())


def make_cache_key(namespace, payload):
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return f"{namespace}:{digest}"


def get_cache(key):
    item = _cache.get(key)

    if not item:
        return None

    expires_at, value = item

    if expires_at < time.time():
        _cache.pop(key, None)
        return None

    return copy.deepcopy(value)


def set_cache(key, value, ttl_seconds=DEFAULT_TTL_SECONDS):
    _cache[key] = (
        time.time() + ttl_seconds,
        copy.deepcopy(value)
    )


def clear_cache():
    # 메모리에 쌓인 캐시(동일 요청 재사용분)를 전부 비운다.
    # 비운 항목 수를 반환해 호출 측에서 확인할 수 있게 한다.
    cleared = len(_cache)
    _cache.clear()

    return cleared


def get_or_set_cache(key, factory, ttl_seconds=DEFAULT_TTL_SECONDS):
    cached_value = get_cache(key)

    if cached_value is not None:
        return cached_value, True

    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())

    with lock:
        cached_value = get_cache(key)

        if cached_value is not None:
            return cached_value, True

        value = factory()
        set_cache(key, value, ttl_seconds)

    return copy.deepcopy(value), False
