"""OpenAI 클라이언트, 재시도, 스트리밍 호출."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"
MAX_RETRIES = 3  # 최초 호출 실패 후 재시도 횟수
BACKOFF_BASE_SECONDS = 1.0  # 1s → 2s → 4s
MAX_RETRY_AFTER = 30.0  # 서버가 요청한 대기 시간의 상한(초)

RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

T = TypeVar("T")


def get_config(key: str, default: str | None = None) -> str | None:
    """st.secrets(Streamlit Cloud) → .env/환경변수(로컬) 순서로 설정값을 읽는다."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # secrets.toml이 없는 로컬 환경
    value = os.getenv(key)
    return value if value else default


def get_api_key() -> str | None:
    return get_config("OPENAI_API_KEY")


def get_model() -> str:
    return get_config("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    # 재시도는 _with_retry에서 직접 관리한다.
    return OpenAI(api_key=api_key, max_retries=0)


def _require_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return get_client(api_key)


def _is_retryable(e: Exception) -> bool:
    if not isinstance(e, RETRYABLE_ERRORS):
        return False
    # 크레딧 소진은 기다려도 해결되지 않는다.
    return getattr(e, "code", None) != "insufficient_quota"


def _retry_delay(e: Exception, attempt: int) -> float:
    """서버가 Retry-After를 주면 그 값을(최대 MAX_RETRY_AFTER초), 없으면 지수 백오프를 쓴다."""
    response = getattr(e, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after-ms", "retry-after"):
        value = headers.get(key)
        if value is None:
            continue
        try:
            seconds = float(value) / (1000 if key == "retry-after-ms" else 1)
        except ValueError:
            continue
        if 0 <= seconds:
            return min(seconds, MAX_RETRY_AFTER)
    return BACKOFF_BASE_SECONDS * 2**attempt


def _with_retry(fn: Callable[[], T]) -> T:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES or not _is_retryable(e):
                raise
            time.sleep(_retry_delay(e, attempt))
    raise AssertionError("unreachable")


def stream_chat(messages: list[dict], model: str | None = None) -> Iterator[str]:
    """응답을 텍스트 조각 단위로 내보내는 제너레이터. st.write_stream()에 바로 전달한다.

    연결 수립(첫 응답 전) 단계의 일시적 오류만 재시도한다.
    """
    client = _require_client()
    stream = _with_retry(
        lambda: client.chat.completions.create(
            model=model or get_model(), messages=messages, stream=True
        )
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def complete(messages: list[dict], model: str | None = None) -> str:
    """스트리밍 없이 전체 응답을 반환한다 (청크 요약 등 내부 처리용)."""
    client = _require_client()
    resp = _with_retry(
        lambda: client.chat.completions.create(model=model or get_model(), messages=messages)
    )
    return resp.choices[0].message.content or ""


def friendly_error(e: Exception) -> str:
    """예외를 사용자에게 보여줄 한국어 메시지로 바꾼다."""
    if isinstance(e, NotFoundError):
        return f"모델 `{get_model()}`을(를) 찾을 수 없습니다. OPENAI_MODEL 설정을 확인하세요."
    if isinstance(e, AuthenticationError):
        return "API 키가 올바르지 않습니다. OPENAI_API_KEY 설정을 확인하세요."
    if isinstance(e, PermissionDeniedError):
        return "이 API 키로는 해당 모델에 접근할 권한이 없습니다."
    if isinstance(e, RateLimitError):
        if getattr(e, "code", None) == "insufficient_quota":
            return "OpenAI 크레딧(사용 한도)이 소진되었습니다. 결제/사용량 설정을 확인하세요."
        return "요청이 너무 많습니다. 잠시 후 다시 시도하세요."
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하고 다시 시도하세요."
    if isinstance(e, InternalServerError):
        return "OpenAI 서버 오류입니다. 잠시 후 다시 시도하세요."
    if isinstance(e, OpenAIError):
        return f"OpenAI API 오류: {e}"
    return f"처리 중 오류가 발생했습니다: {e}"
