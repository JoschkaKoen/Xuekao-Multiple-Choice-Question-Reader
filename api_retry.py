"""Retry/backoff for API calls — trimmed from eXercise/api_retry.py.

Fail-open: retry anything that isn't clearly terminal (4xx auth/bad-request,
cancellation, JSON decode). DashScope wraps transient server flakes in HTTP 400
with an ``InternalError.Algo`` body — those are retried.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger("course_reader")

_NEVER_RETRY = (KeyboardInterrupt, SystemExit, GeneratorExit, json.JSONDecodeError)


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, _NEVER_RETRY):
        return False
    try:
        from openai import APIStatusError
        if isinstance(exc, APIStatusError):
            dashscope_transient_400 = (
                exc.status_code == 400 and "InternalError.Algo" in str(exc)
            )
            if not dashscope_transient_400 and exc.status_code in (400, 401, 403, 404, 422):
                return False
    except ImportError:
        pass
    return True


def retry_api_call(
    fn: Callable[[], T],
    *,
    label: str,
    max_attempts: int = 4,
    base_sleep: float = 0.1,
    backoff_factor: float = 2.0,
    max_sleep: float = 5.0,
    jitter: float = 0.25,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:
            if attempt >= max_attempts or not is_retryable(exc):
                raise
            sleep_s = min(base_sleep * (backoff_factor ** (attempt - 1)), max_sleep)
            if jitter:
                sleep_s *= random.uniform(1.0 - jitter, 1.0 + jitter)
            log.info(
                "%s: API error (attempt %d/%d) — %s; retrying in %.2fs",
                label, attempt, max_attempts, exc, sleep_s,
            )
            time.sleep(sleep_s)
