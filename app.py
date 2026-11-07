"""PDF 분석기 — Streamlit 진입점."""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import datetime

import streamlit as st

from analyzer import (
    analysis_messages,
    build_context,
    chat_messages,
    docs_to_condense,
    history_chars,
    needs_condensing,
)
from llm import friendly_error, get_api_key, get_config, get_model, stream_chat
from pdf_utils import EncryptedPdfError, InvalidPdfError, PdfDocument, extract_pages
from prompts import ANALYSIS_TYPES, COMPARE, CUSTOM, LANGUAGES, SUMMARY_LENGTHS

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

# 색상은 테마(config.toml)에 맡기고, 간격·크기만 조정한다 → 라이트/다크 모두 대응
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1200px; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
    [data-testid="stSidebar"] h3 { font-size: 1rem; margin-bottom: 0; }
    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.6rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TYPE_ICONS = {
    "요약": ":material/summarize:",
    "핵심 포인트": ":material/format_list_bulleted:",
    "키워드/개체": ":material/sell:",
    "문서 비교": ":material/compare:",
    "사용자 정의": ":material/edit_note:",
}
MAX_UPLOAD_MB = 50  # .streamlit/config.toml의 maxUploadSize와 맞출 것


# ---------------------------------------------------------------- 접근 제어
def require_password() -> None:
    """APP_PASSWORD가 설정되어 있으면 비밀번호를 입력해야 앱을 쓸 수 있다.

    공개 배포 시 누구나 API 크레딧을 소모하는 것을 막는다. 설정하지 않으면(로컬 개발) 생략.
    """
    expected = get_config("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return

    st.title("📄 PDF 분석기")
    with st.form("login"):
        password = st.text_input("비밀번호", type="password", placeholder="관리자에게 받은 비밀번호")
        submitted = st.form_submit_button("들어가기", type="primary", icon=":material/login:")
    if submitted:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.", icon=":material/lock:")
    st.stop()


# ---------------------------------------------------------------- 상태 관리
def init_state() -> None:
    defaults = {
        "doc_fingerprint": (),
        "analysis": None,  # {"type", "content", "docs", "model", "created"} — content는 원문
        "chat_history": [],  # [{"role", "content"}] — content는 원문
        "condensed": {},  # doc_id -> 청크 요약본
        "chat_pending": False,  # 답변 생성 중이면 True (중단 감지용)
        "analysis_running": None,  # 진행 중인 분석 유형 (중단 감지용)
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def recover_interrupted_runs() -> tuple[str | None, str | None]:
    """이전 실행이 응답 도중 중단됐는지 확인하고 상태를 정리한다.

    Streamlit은 새 입력이 들어오면 실행 중인 스크립트를 멈추고 다시 실행하므로,
    스트리밍 중이던 답변·분석은 저장되지 못하고 사라진다.
    반환: (채팅 안내 문구, 분석 안내 문구)
    """
    chat_notice = analysis_notice = None
    if st.session_state.chat_pending:
        history = st.session_state.chat_history
        if history and history[-1]["role"] == "user":
            dropped = history.pop()
            short = dropped["content"][:40] + ("…" if len(dropped["content"]) > 40 else "")
            chat_notice = f"답변 생성 중에 다른 입력이 들어와 '{short}' 질문의 답변이 중단되었습니다. 필요하면 다시 질문하세요."
        st.session_state.chat_pending = False
    if st.session_state.analysis_running:
        analysis_notice = (
            f"'{st.session_state.analysis_running}' 분석이 진행 중에 설정 변경 등으로 중단되었습니다. "
            "분석이 끝날 때까지 기다린 뒤 설정을 바꾸고, 다시 [분석 시작]을 눌러 주세요."
        )
        st.session_state.analysis_running = None
    return chat_notice, analysis_notice


def reset_conversation() -> None:
    st.session_state.chat_history = []
    st.session_state.chat_pending = False
    st.toast("대화 기록을 초기화했습니다.", icon=":material/restart_alt:")


def on_documents_changed(fingerprint: tuple) -> None:
    if st.session_state.doc_fingerprint != fingerprint:
        st.session_state.doc_fingerprint = fingerprint
        st.session_state.analysis = None
        st.session_state.chat_history = []


# ---------------------------------------------------------------- 문서 로딩
@st.cache_data(show_spinner=False, max_entries=20)
def cached_extract(file_bytes: bytes) -> list[dict]:
    return extract_pages(file_bytes)


def load_documents(uploaded_files) -> tuple[list[PdfDocument], list[tuple[str, str]]]:
    """업로드 파일을 문서로 변환한다. 반환: (문서 목록, [(수준 "warning"|"info", 메시지)])"""
    docs: list[PdfDocument] = []
    notices: list[tuple[str, str]] = []
    seen: dict[str, str] = {}  # doc_id -> 처음 올린 파일명
    for f in uploaded_files:
        data = f.getvalue()
        doc_id = hashlib.sha256(data).hexdigest()
        if doc_id in seen:
            notices.append(("info", f"**{f.name}** — '{seen[doc_id]}'와 내용이 같은 파일이라 한 번만 분석합니다."))
            continue
        seen[doc_id] = f.name
        try:
            with st.spinner(f"{f.name} 텍스트 추출 중..."):
                pages = cached_extract(data)
        except EncryptedPdfError:
            notices.append(("warning", f"**{f.name}** — 암호가 걸린 PDF는 지원하지 않습니다."))
            continue
        except InvalidPdfError:
            notices.append(("warning", f"**{f.name}** — PDF 파일을 읽을 수 없습니다. 손상되었거나 PDF 형식이 아닙니다."))
            continue
        except Exception:  # 한 파일의 예기치 못한 오류가 앱 전체를 멈추지 않도록
            notices.append(("warning", f"**{f.name}** — 텍스트를 추출하는 중 알 수 없는 오류가 발생했습니다."))
            continue

        doc = PdfDocument(name=f.name, pages=pages, doc_id=doc_id)
        if doc.is_scanned:
            notices.append(("warning", f"**{f.name}** — 스캔 이미지 PDF로 보입니다. 텍스트 추출이 불가합니다."))
            continue
        docs.append(doc)
    return docs, notices


def get_context(docs: list[PdfDocument], status=None, reserved_chars: int = 0) -> str:
    """문서 컨텍스트를 만든다. 대용량이면 진행률을 표시하며 청크 요약을 수행한다."""
    cache = st.session_state.condensed
    if not needs_condensing(docs, reserved_chars):
        return build_context(docs, cache, reserved_chars=reserved_chars)
    if status is not None:
        status.write("긴 문서를 나누어 요약하는 중...")
    bar = st.progress(0.0, text="긴 문서를 나누어 요약하는 중...")
    try:
        return build_context(
            docs, cache, lambda pct, msg: bar.progress(pct, text=msg), reserved_chars=reserved_chars
        )
    finally:
        bar.empty()


# ---------------------------------------------------------------- 텍스트 처리
def escape_markdown_tilde(text: str) -> str:
    """'~'를 이스케이프한다.

    Streamlit Markdown은 ~a~ 를 취소선으로 렌더링하는데, 한국어 답변은 범위 표기
    (p.4~5, 0.5~0.9ppm)에 '~'를 많이 써서 문장 전체가 취소선 처리되는 문제가 있다.
    저장은 원문으로 하고, 화면에 표시할 때만 이 함수를 적용한다.
    """
    return text.replace("~", "\\~")


def unescape_markdown_tilde(text: str) -> str:
    return text.replace("\\~", "~")


def wait_first_chunk(stream, message: str):
    """첫 응답 조각이 올 때까지 스피너를 보여준 뒤, 나머지를 이어서 내보낸다.

    모델이 첫 토큰을 내기까지 수십 초 걸릴 수 있어 빈 화면을 방지한다.
    표시용 이스케이프를 적용하므로, st.write_stream의 반환값은 unescape해서 저장한다.
    """
    with st.spinner(message):
        first = next(stream, None)

    def gen():
        if first is not None:
            yield escape_markdown_tilde(first)
        for chunk in stream:
            yield escape_markdown_tilde(chunk)

    return gen()


def stream_to_text(written) -> str:
    """st.write_stream 반환값(이스케이프된 표시용 텍스트)을 원문으로 되돌린다."""
    text = written if isinstance(written, str) else "".join(map(str, written))
    return unescape_markdown_tilde(text)


def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", text)


def export_text(analysis: dict, markdown: bool) -> str:
    """다운로드용 텍스트. Markdown 파일은 '~'를 이스케이프하고, 텍스트 파일은 원문 그대로."""
    content = escape_markdown_tilde(analysis["content"]) if markdown else analysis["content"]
    doc_list = "\n".join(f"- {name}" for name in analysis["docs"])
    title = "# " if markdown else ""
    section = "## " if markdown else "■ "
    return (
        f"{title}PDF 분석 결과 — {analysis['type']}\n\n"
        f"- 생성 시각: {analysis['created']}\n- 모델: {analysis['model']}\n\n"
        f"{section}대상 문서\n{doc_list}\n\n{section}결과\n\n{content}\n"
    )


# ---------------------------------------------------------------- 화면 조각
def render_header() -> None:
    st.title("📄 PDF 분석기")
    st.caption("PDF를 업로드하면 AI가 문서를 읽고 요약·분석하며, 문서 내용에 대한 질문에 답합니다.")
    st.badge(f"모델: {get_model()}", icon=":material/smart_toy:", color="violet")


def render_onboarding() -> None:
    st.markdown("#### 시작하기")
    steps = [
        (":material/upload_file:", "1. PDF 업로드", "왼쪽 사이드바에 PDF 파일을 끌어다 놓으세요. 여러 개를 함께 올릴 수 있습니다."),
        (":material/tune:", "2. 분석 유형 선택", "요약, 핵심 포인트, 키워드/개체, 문서 비교, 사용자 정의 중에서 고르세요."),
        (":material/play_circle:", "3. 분석 시작", "[분석 시작]을 누르거나, 'Q&A 채팅' 탭에서 문서에 대해 질문하세요."),
    ]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, steps):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
    st.caption(
        f"※ 텍스트가 있는 PDF만 분석할 수 있습니다 (스캔 이미지·암호 PDF 제외, 파일당 최대 {MAX_UPLOAD_MB}MB). "
        "업로드한 파일은 서버에 저장되지 않습니다."
    )


def render_documents(docs: list[PdfDocument]) -> None:
    condensed_ids = docs_to_condense(docs)
    st.markdown(f"#### 업로드된 문서 ({len(docs)}개)")
    per_row = 3
    for i in range(0, len(docs), per_row):
        cols = st.columns(per_row)
        for col, doc in zip(cols, docs[i : i + per_row]):
            with col.container(border=True):
                st.markdown(f"**:material/description: {escape_markdown_tilde(doc.name)}**")
                m1, m2 = st.columns(2)
                m1.metric("페이지", f"{doc.page_count:,}")
                m2.metric("글자 수", f"{doc.char_count:,}")
                if doc.doc_id in condensed_ids:
                    st.caption(":material/compress: 분량이 많아 청크 요약 후 분석합니다.")
                with st.expander("텍스트 미리보기"):
                    st.text(doc.full_text()[:3000])


def render_analysis_tab(docs, run, analysis_type, language, summary_length, custom_prompt, notice) -> None:
    if notice and not run:
        st.warning(notice, icon=":material/pause_circle:")

    if run:
        st.session_state.analysis_running = analysis_type
        with st.status(f"'{analysis_type}' 분석 준비 중...", expanded=True) as status:
            try:
                status.write("문서 내용을 정리하는 중...")
                context = get_context(docs, status)
                status.update(label=f"'{analysis_type}' 분석 결과 생성 중...", state="running")
            except Exception as e:
                st.session_state.analysis_running = None
                status.update(label="분석 실패", state="error")
                st.error(friendly_error(e), icon=":material/error:")
                return
        with st.container(border=True):
            try:
                messages = analysis_messages(context, analysis_type, language, summary_length, custom_prompt)
                written = st.write_stream(
                    wait_first_chunk(stream_chat(messages), "AI가 문서를 분석하고 있습니다...")
                )
            except Exception as e:
                st.session_state.analysis_running = None
                status.update(label="분석 실패", state="error", expanded=False)
                st.error(friendly_error(e), icon=":material/error:")
                return
        st.session_state.analysis_running = None
        status.update(label=f"'{analysis_type}' 분석 완료", state="complete", expanded=False)
        st.session_state.analysis = {
            "type": analysis_type,
            "content": stream_to_text(written),
            "docs": [d.name for d in docs],
            "model": get_model(),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.toast("분석이 완료되었습니다.", icon=":material/check_circle:")
    elif st.session_state.analysis:
        a = st.session_state.analysis
        st.caption(f"{TYPE_ICONS.get(a['type'], '')} {a['type']} · {a['created']} · {a['model']}")
        with st.container(border=True):
            st.markdown(escape_markdown_tilde(a["content"]))
    else:
        st.info(
            "사이드바에서 분석 유형을 고른 뒤 **[분석 시작]** 버튼을 누르면 결과가 여기에 표시됩니다.",
            icon=":material/info:",
        )

    if st.session_state.analysis:
        a = st.session_state.analysis
        base = f"pdf_analysis_{safe_filename(a['type'])}"
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button(
            "결과 다운로드 (.md)",
            data=export_text(a, markdown=True),
            file_name=f"{base}.md",
            mime="text/markdown",
            icon=":material/download:",
            help="Markdown 뷰어(GitHub, Notion, Obsidian 등)용",
            width="stretch",
        )
        c2.download_button(
            "텍스트로 다운로드 (.txt)",
            data=export_text(a, markdown=False),
            file_name=f"{base}.txt",
            mime="text/plain",
            icon=":material/description:",
            help="메모장 등 일반 텍스트 편집기용",
            width="stretch",
        )


def render_chat_tab(docs, language, api_key, notice) -> None:
    messages_area = st.container()
    question = st.chat_input(
        "이 문서의 결론은 무엇인가요?" if api_key else "API 키를 설정하면 질문할 수 있습니다",
        disabled=not api_key,
    )

    with messages_area:
        if notice:
            st.warning(notice, icon=":material/pause_circle:")
        if not st.session_state.chat_history and not question:
            st.caption(
                ":material/forum: 문서 내용에 대해 자유롭게 질문하세요. "
                "답변에는 근거 페이지가 (p.N) 형식으로 표시됩니다. "
                "답변이 끝난 뒤 다음 질문을 입력해 주세요."
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(escape_markdown_tilde(msg["content"]))

        if question:
            history = st.session_state.chat_history
            history.append({"role": "user", "content": question})
            st.session_state.chat_pending = True
            with st.chat_message("user"):
                st.markdown(escape_markdown_tilde(question))
            with st.chat_message("assistant"):
                try:
                    context = get_context(docs, reserved_chars=history_chars(history))
                    stream = stream_chat(chat_messages(context, language, history))
                    written = st.write_stream(wait_first_chunk(stream, "답변을 준비하는 중..."))
                # finally를 쓰지 않는다: Streamlit의 재실행 중단(RerunException, BaseException 계열)
                # 때도 실행되어 중단 감지 플래그가 지워지기 때문이다.
                except Exception as e:
                    history.pop()  # 실패한 질문은 기록에서 제외
                    st.session_state.chat_pending = False
                    st.error(friendly_error(e), icon=":material/error:")
                else:
                    history.append({"role": "assistant", "content": stream_to_text(written)})
                    st.session_state.chat_pending = False


# ---------------------------------------------------------------- 메인
require_password()
init_state()
chat_notice, analysis_notice = recover_interrupted_runs()
api_key = get_api_key()

with st.sidebar:
    st.subheader("① PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=f"여러 파일을 한 번에 올릴 수 있습니다. 파일당 최대 {MAX_UPLOAD_MB}MB.",
    )

documents, load_notices = load_documents(uploaded_files or [])
on_documents_changed(tuple(d.doc_id for d in documents))

with st.sidebar:
    st.divider()
    st.subheader("② 분석 설정")
    available_types = [t for t in ANALYSIS_TYPES if t != COMPARE or len(documents) >= 2]
    analysis_type = st.radio(
        "분석 유형",
        available_types,
        format_func=lambda t: f"{TYPE_ICONS[t]} {t}",
    )
    if len(documents) < 2:
        st.caption(":material/info: '문서 비교'는 PDF를 2개 이상 올리면 나타납니다.")

    summary_length = "보통"
    if analysis_type == "요약":
        summary_length = st.segmented_control(
            "요약 길이", list(SUMMARY_LENGTHS), default="보통", required=True
        ) or "보통"

    custom_prompt = ""
    if analysis_type == CUSTOM:
        custom_prompt = st.text_area(
            "분석 지시문", placeholder="예: 이 문서의 리스크 요인을 표로 정리해줘", height=100
        )

    language = st.segmented_control(
        "출력 언어", list(LANGUAGES), default="한국어", required=True
    ) or "한국어"

    st.divider()
    st.subheader("③ 실행")
    disabled_reason = None
    if not api_key:
        disabled_reason = "OPENAI_API_KEY가 설정되지 않았습니다."
    elif not documents:
        disabled_reason = "분석할 PDF를 먼저 업로드하세요."
    elif analysis_type == CUSTOM and not custom_prompt.strip():
        disabled_reason = "분석 지시문을 입력하세요."
    run_analysis = st.button(
        "분석 시작",
        type="primary",
        icon=":material/play_arrow:",
        disabled=disabled_reason is not None,
        help=disabled_reason,
        width="stretch",
    )
    st.caption(":material/schedule: 분석·답변이 끝날 때까지 설정을 바꾸지 마세요. 진행 중인 작업이 중단됩니다.")
    # 채팅 처리 후 기록 유무를 반영하도록 맨 마지막에 그린다.
    reset_slot = st.empty()


def render_reset_button() -> None:
    has_history = bool(st.session_state.chat_history)
    reset_slot.button(
        "대화 초기화",
        icon=":material/restart_alt:",
        on_click=reset_conversation,
        disabled=not has_history,
        help=None if has_history else "초기화할 대화가 없습니다.",
        width="stretch",
    )


render_header()

if not api_key:
    st.error(
        "**OPENAI_API_KEY가 설정되지 않았습니다.**\n\n"
        "- 로컬: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=...`를 추가하세요.\n"
        "- Streamlit Cloud: 앱 Settings → Secrets에 `OPENAI_API_KEY`를 입력하세요.",
        icon=":material/key_off:",
    )

for level, msg in load_notices:
    if level == "info":
        st.info(msg, icon=":material/content_copy:")
    else:
        st.warning(msg, icon=":material/warning:")

if not documents:
    render_onboarding()
    render_reset_button()
    st.stop()

render_documents(documents)

tab_result, tab_chat = st.tabs([":material/analytics: 분석 결과", ":material/chat: Q&A 채팅"])
with tab_result:
    render_analysis_tab(
        documents, run_analysis, analysis_type, language, summary_length, custom_prompt, analysis_notice
    )
with tab_chat:
    render_chat_tab(documents, language, api_key, chat_notice)

render_reset_button()
