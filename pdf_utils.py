"""PDF 텍스트 추출 유틸리티."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import DependencyError

# 추출된 글자 수가 이 값보다 적으면 스캔본(이미지 PDF)으로 간주한다.
MIN_TEXT_CHARS = 20


class PdfExtractionError(Exception):
    """PDF를 읽을 수 없을 때 발생한다."""


class EncryptedPdfError(PdfExtractionError):
    """암호가 걸린 PDF."""


class InvalidPdfError(PdfExtractionError):
    """PDF 형식이 아니거나 손상된 파일."""


@dataclass
class PdfDocument:
    name: str
    pages: list[dict] = field(default_factory=list)  # [{"page": 1, "text": "..."}]
    doc_id: str = ""  # 파일 내용 해시 (문서 변경 감지·요약 캐시 키)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p["text"]) for p in self.pages)

    @property
    def is_scanned(self) -> bool:
        """텍스트가 거의 없으면 스캔 이미지 PDF로 본다."""
        return self.char_count < MIN_TEXT_CHARS

    def full_text(self) -> str:
        return "\n".join(f"[p.{p['page']}] {p['text']}" for p in self.pages if p["text"])


def split_into_chunks(pages: list[dict], chunk_chars: int) -> list[dict]:
    """페이지 경계를 지키며 chunk_chars 이하 크기의 청크로 나눈다.

    한 페이지가 chunk_chars보다 길면 그 페이지만 글자 수 기준으로 자른다.
    반환: [{"start": 첫 페이지, "end": 끝 페이지, "text": "[p.N] ..."}]
    """
    chunks: list[dict] = []
    buf: list[str] = []
    start = end = None
    size = 0

    def flush():
        nonlocal buf, start, end, size
        if buf:
            chunks.append({"start": start, "end": end, "text": "\n".join(buf)})
        buf, start, end, size = [], None, None, 0

    for p in pages:
        if not p["text"]:
            continue
        line = f"[p.{p['page']}] {p['text']}"
        if len(line) > chunk_chars:
            flush()
            for i in range(0, len(line), chunk_chars):
                chunks.append({"start": p["page"], "end": p["page"], "text": line[i : i + chunk_chars]})
            continue
        if size + len(line) > chunk_chars:
            flush()
        if start is None:
            start = p["page"]
        end = p["page"]
        buf.append(line)
        size += len(line) + 1
    flush()
    return chunks


def _clean(text: str) -> str:
    # 일부 PDF는 공백 대신 NUL(\x00)을 쓴다.
    return text.replace("\x00", " ").strip()


def extract_pages(file_bytes: bytes) -> list[dict]:
    """PDF 바이트에서 페이지별 텍스트를 추출한다.

    bytes를 입력으로 받으므로 st.cache_data로 감쌀 수 있다.

    Raises:
        EncryptedPdfError: 암호화된 PDF
        InvalidPdfError: PDF가 아니거나 손상된 파일
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except DependencyError as e:
        # AES 암호화 PDF는 열 때 바로 복호화를 시도하므로 여기서 실패할 수 있다.
        raise EncryptedPdfError("이 PDF의 암호화 방식을 처리할 수 없습니다.") from e
    except Exception as e:  # pypdf는 손상 파일에 다양한 예외를 던진다
        raise InvalidPdfError(str(e)) from e

    if reader.is_encrypted:
        # 빈 비밀번호로 열리는 PDF(권한 제한만 걸린 경우)는 허용한다.
        try:
            if not reader.decrypt(""):
                raise EncryptedPdfError("암호가 걸린 PDF입니다.")
        except EncryptedPdfError:
            raise
        except Exception as e:
            raise EncryptedPdfError("암호가 걸린 PDF입니다.") from e

    pages: list[dict] = []
    try:
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append({"page": i, "text": _clean(text)})
    except Exception as e:
        raise InvalidPdfError(str(e)) from e

    return pages
