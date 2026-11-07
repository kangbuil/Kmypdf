"""문서 컨텍스트 구성 및 분석 메시지 생성 (대용량 문서는 Map-Reduce)."""

from __future__ import annotations

from collections.abc import Callable

from llm import complete, get_config
from pdf_utils import PdfDocument, split_into_chunks
from prompts import analysis_instruction, chunk_summary_messages, format_document, system_prompt

def _int_config(key: str, default: int) -> int:
    """정수 설정값을 읽는다. 값이 잘못되면 기본값을 쓴다."""
    try:
        value = int(get_config(key, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


# 전체 입력(문서 + 대화 기록) 글자 수가 이 값을 넘으면 큰 문서부터 청크 요약본으로 대체한다.
MAX_CONTEXT_CHARS = _int_config("MAX_CONTEXT_CHARS", 120_000)
# Map 단계 청크 크기(글자 수)
CHUNK_CHARS = _int_config("CHUNK_CHARS", 30_000)
# 채팅에서 모델에 보내는 최근 대화 턴 수 (질문+답변 = 1턴)
MAX_CHAT_TURNS = _int_config("MAX_CHAT_TURNS", 10)

ProgressFn = Callable[[float, str], None]


def condense_document(doc: PdfDocument, on_progress: ProgressFn | None = None) -> str:
    """Map 단계: 문서를 청크로 나눠 각각 요약한 뒤 이어 붙인다."""
    chunks = split_into_chunks(doc.pages, CHUNK_CHARS)
    parts = []
    for i, c in enumerate(chunks):
        if on_progress:
            on_progress(i / len(chunks), f"'{doc.name}' 청크 요약 중 ({i + 1}/{len(chunks)})")
        summary = complete(chunk_summary_messages(doc.name, c["start"], c["end"], c["text"]))
        parts.append(f"[요약: p.{c['start']}~p.{c['end']}]\n{summary}")
    if on_progress:
        on_progress(1.0, f"'{doc.name}' 요약 완료")
    return "\n\n".join(parts)


def needs_condensing(docs: list[PdfDocument], reserved_chars: int = 0) -> bool:
    return sum(d.char_count for d in docs) + reserved_chars > MAX_CONTEXT_CHARS


def docs_to_condense(docs: list[PdfDocument]) -> set[str]:
    """한도에 맞추기 위해 요약본으로 대체할 문서의 doc_id (큰 문서부터).

    원문 길이 기준 추정치이며, 실제 대체 여부는 build_context가 요약본 길이로 판단한다.
    """
    total = sum(d.char_count for d in docs)
    ids: set[str] = set()
    for d in sorted(docs, key=lambda d: d.char_count, reverse=True):
        if total <= MAX_CONTEXT_CHARS:
            break
        ids.add(d.doc_id)
        total -= d.char_count
    return ids


def build_context(
    docs: list[PdfDocument],
    condensed_cache: dict[str, str],
    on_progress: ProgressFn | None = None,
    reserved_chars: int = 0,
) -> str:
    """모델에 넘길 <document> 블록들을 만든다.

    문서 길이 + reserved_chars(채팅 기록 등)가 MAX_CONTEXT_CHARS를 넘으면
    큰 문서부터 요약본으로 바꿔 한도 안에 맞춘다.
    요약본은 condensed_cache(doc_id → 요약)에 저장해 재사용한다.
    """
    bodies = {d.doc_id: d.full_text() for d in docs}
    total = sum(len(b) for b in bodies.values()) + reserved_chars

    for d in sorted(docs, key=lambda d: d.char_count, reverse=True):
        if total <= MAX_CONTEXT_CHARS:
            break
        if d.doc_id not in condensed_cache:
            condensed_cache[d.doc_id] = condense_document(d, on_progress)
        total -= len(bodies[d.doc_id])
        bodies[d.doc_id] = condensed_cache[d.doc_id]
        total += len(bodies[d.doc_id])

    return "\n\n".join(format_document(d.name, bodies[d.doc_id]) for d in docs)


def analysis_messages(
    context: str,
    analysis_type: str,
    language: str,
    summary_length: str = "보통",
    custom: str = "",
) -> list[dict]:
    """Reduce 단계(또는 단일 호출): 분석 요청 메시지."""
    instruction = analysis_instruction(analysis_type, summary_length, custom)
    return [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": f"{context}\n\n[분석 요청]\n{instruction}"},
    ]


def recent_history(history: list[dict]) -> list[dict]:
    """모델에 보낼 최근 MAX_CHAT_TURNS턴만 남긴다 (항상 user 메시지로 시작)."""
    recent = history[-MAX_CHAT_TURNS * 2 :]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return recent


def history_chars(history: list[dict]) -> int:
    return sum(len(m["content"]) for m in recent_history(history))


def chat_messages(context: str, language: str, history: list[dict]) -> list[dict]:
    """Q&A: 문서 컨텍스트 + 최근 대화.

    문서 컨텍스트를 앞쪽 고정 위치에 두어, 모델 제공자의 자동 프롬프트 캐싱
    (동일한 앞부분 재사용)이 적용될 수 있게 한다.
    """
    return [
        {"role": "system", "content": system_prompt(language)},
        {
            "role": "user",
            "content": f"{context}\n\n위 문서를 바탕으로 이어지는 질문에 답해주세요.",
        },
        {"role": "assistant", "content": "네, 문서 내용을 바탕으로 답변하겠습니다."},
        *recent_history(history),
    ]
