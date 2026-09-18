import os
import re
import json
import logging

import pymupdf as fitz  # PyMuPDF — `import fitz` directly is deprecated, `import pymupdf as fitz` is the current form
import pymupdf4llm
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pdf2image import convert_from_path
import pytesseract

from common import new_id, now_iso
from experience import compute_total_years
from title_normalization import split_flattened_company_title

logger = logging.getLogger("talentlens.extractor")

# Setting the data source to read directly from a local Kaggle directory path
DATA_DIR = os.path.expanduser("~/kaggle/input/talentlens-batch/")

MIN_TEXT_DENSITY_CHARS_PER_PAGE = 40  # below this, assume scanned/image PDF, fall back to OCR


# --- Stage 1: deterministic extraction (no LLM) ---

def _iter_block_items(doc: Document):
    """
    Yields paragraphs and tables in true document order. python-docx's .paragraphs and
    .tables are separate flat lists with no ordering between them, so a resume with e.g.
    a skills table in the middle of prose would come out with the table's content moved
    to the end — this walks the underlying XML body directly to preserve real order.
    """
    for child in doc.element.body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, doc)
        elif child.tag == qn('w:tbl'):
            yield Table(child, doc)


def _docx_to_markdown(file_path: str) -> str:
    """Structure-aware DOCX -> markdown: headings become #, list items become -, tables
    become markdown tables — all exact text, no summarization or rephrasing."""
    doc = Document(file_path)
    lines = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style_name = (block.style.name or "").lower()
            if "heading" in style_name or "title" in style_name:
                level = next((int(ch) for ch in style_name if ch.isdigit()), 1)
                lines.append(f"{'#' * min(max(level, 1), 6)} {text}")
            elif "list" in style_name:
                lines.append(f"- {text}")
            else:
                lines.append(text)

        elif isinstance(block, Table):
            rows = block.rows
            if not rows:
                continue
            header = [c.text.strip() for c in rows[0].cells]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                cells = [c.text.strip() for c in row.cells]
                lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# --- Stage 1 addition: column-aware page reordering for multi-column PDFs ---

_FULL_WIDTH_FRACTION = 0.6        # block wider than this fraction of page width = header/divider, not a column
_MIN_COLUMN_BLOCKS = 4            # fewer candidate blocks than this = not enough evidence to call it two-column
_MIN_COLUMN_GAP_FRACTION = 0.03   # required horizontal gap between the two x-clusters, as a fraction of page width
_MIN_COLUMN_HEIGHT_OVERLAP = 0.3  # left/right block y-ranges must overlap by at least this fraction of page height,
                                   # so a real running sidebar is required — not just one late block with a different x


def _cluster_columns_1d(xs: list) -> tuple | None:
    """Simple 1D two-means clustering on block left-x (x0) positions."""
    if len(xs) < _MIN_COLUMN_BLOCKS:
        return None

    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-6:
        return None

    left_c, right_c = lo, hi
    for _ in range(20):
        left_pts = [x for x in xs if abs(x - left_c) <= abs(x - right_c)]
        right_pts = [x for x in xs if abs(x - left_c) > abs(x - right_c)]
        if not left_pts or not right_pts:
            return None  # collapsed to one cluster -- not two-column
        new_left = sum(left_pts) / len(left_pts)
        new_right = sum(right_pts) / len(right_pts)
        converged = abs(new_left - left_c) < 0.5 and abs(new_right - right_c) < 0.5
        left_c, right_c = new_left, new_right
        if converged:
            break

    return (left_c, right_c) if left_c < right_c else (right_c, left_c)


def _page_two_column_split_x(narrow_blocks: list, page_width: float, page_height: float):
    """Returns the x-coordinate to split narrow_blocks into left/right columns."""
    if len(narrow_blocks) < _MIN_COLUMN_BLOCKS:
        return None

    centers = _cluster_columns_1d([b[0] for b in narrow_blocks])
    if centers is None:
        return None
    left_c, right_c = centers

    if (right_c - left_c) < _MIN_COLUMN_GAP_FRACTION * page_width:
        return None

    split_x = (left_c + right_c) / 2
    left_blocks = [b for b in narrow_blocks if b[0] < split_x]
    right_blocks = [b for b in narrow_blocks if b[0] >= split_x]
    if len(left_blocks) < 2 or len(right_blocks) < 2:
        return None

    left_y_span = (min(b[1] for b in left_blocks), max(b[3] for b in left_blocks))
    right_y_span = (min(b[1] for b in right_blocks), max(b[3] for b in right_blocks))
    overlap = min(left_y_span[1], right_y_span[1]) - max(left_y_span[0], right_y_span[0])
    if overlap < _MIN_COLUMN_HEIGHT_OVERLAP * page_height:
        return None

    return split_x


def _reorder_page_columns_aware(page) -> str:
    """Returns column-corrected text for one fitz page."""
    page_width, page_height = page.rect.width, page.rect.height
    raw_blocks = [b for b in page.get_text("blocks") if b[4].strip() and b[6] == 0]

    full_width = [b for b in raw_blocks if (b[2] - b[0]) > _FULL_WIDTH_FRACTION * page_width]
    narrow = [b for b in raw_blocks if (b[2] - b[0]) <= _FULL_WIDTH_FRACTION * page_width]

    split_x = _page_two_column_split_x(narrow, page_width, page_height)
    if split_x is None:
        return page.get_text("text")

    left = sorted((b for b in narrow if b[0] < split_x), key=lambda b: b[1])
    right = sorted((b for b in narrow if b[0] >= split_x), key=lambda b: b[1])

    narrow_top = min(b[1] for b in narrow)
    header = sorted((b for b in full_width if b[1] < narrow_top), key=lambda b: b[1])
    trailer = sorted((b for b in full_width if b[1] >= narrow_top), key=lambda b: b[1])

    ordered = [b[4].strip() for b in header + left + right + trailer]
    return "\n\n".join(t for t in ordered if t)


def _extract_pdf_text_column_aware(file_path: str):
    """Whole-document column-aware extraction."""
    try:
        with fitz.open(file_path) as doc:
            any_two_column = False
            pages_out = []
            for page in doc:
                page_width, page_height = page.rect.width, page.rect.height
                raw_blocks = [b for b in page.get_text("blocks") if b[4].strip() and b[6] == 0]
                narrow = [b for b in raw_blocks if (b[2] - b[0]) <= _FULL_WIDTH_FRACTION * page_width]
                if _page_two_column_split_x(narrow, page_width, page_height) is not None:
                    any_two_column = True
                    pages_out.append(_reorder_page_columns_aware(page))
                else:
                    pages_out.append(page.get_text("text"))
    except Exception as e:
        logger.warning("Column-aware PDF extraction failed, falling back to pymupdf4llm: %s", e)
        return None

    if not any_two_column:
        return None

    return "\n\n".join(pages_out)


def extract_text(file_path: str) -> tuple[str, list[str], str]:
    """
    Extracts exact text from PDF, DOCX, or TXT files as markdown (PDF, DOCX) or plain
    text (TXT) — deterministic, no LLM involved at this stage.
    """
    ext = os.path.splitext(file_path)[1].lower()
    warnings = []

    if ext == '.pdf':
        with fitz.open(file_path) as doc:
            num_pages = doc.page_count

        column_aware_text = _extract_pdf_text_column_aware(file_path)
        if column_aware_text is not None:
            warnings.append(
                "Detected a multi-column page layout (e.g. a sidebar running "
                "alongside a main column) — used column-aware block reordering "
                "instead of pymupdf4llm's default reading order, to avoid "
                "interleaving sidebar content with main-column content."
            )
            raw_text = column_aware_text
        else:
            raw_text = pymupdf4llm.to_markdown(file_path)

        avg_density = len(raw_text) / max(num_pages, 1)

        if avg_density >= MIN_TEXT_DENSITY_CHARS_PER_PAGE:
            return raw_text, warnings, "pdf_markdown"

        warnings.append("Low text density in PDF text layer — used OCR fallback.")
        images = convert_from_path(file_path)
        ocr_parts = [pytesseract.image_to_string(img) for img in images]
        ocr_text = "\n".join(ocr_parts).strip()
        if not ocr_text:
            warnings.append("OCR fallback also produced no text — file may be blank, corrupt, or unreadable.")
        return ocr_text, warnings, "pdf_ocr"

    elif ext == '.docx':
        raw_text = _docx_to_markdown(file_path)
        if not raw_text or not raw_text.strip():
            warnings.append("No extractable text found in DOCX — file may be empty or malformed.")
        return raw_text, warnings, "docx_markdown"

    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text, warnings, "txt"

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# --- Stage 2: LLM structuring only (exact text -> JSON, no arithmetic) ---

def _attempt_json_repair(raw_output: str):
    """Best-effort salvage of a malformed model response, tried only after
    json.loads() has already failed."""
    try:
        from json_repair import repair_json
    except ImportError:
        logger.warning(
            "json_repair package not installed — cannot salvage truncated model output. "
            "Run: pip install json-repair"
        )
        return None

    try:
        repaired = repair_json(raw_output, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except Exception as e:
        logger.warning("json_repair raised while salvaging output: %s", e)
    return None


def _dedupe_repeated_experience(structured: dict) -> int:
    """Collapses experience entries that are exact duplicates."""
    entries = structured.get("experience") or []
    if not entries:
        return 0

    def _key(e):
        return (
            (e.get("title") or "").strip().lower(),
            (e.get("company") or "").strip().lower(),
            (e.get("start_date_raw") or "").strip().lower(),
            (e.get("end_date_raw") or "").strip().lower(),
            re.sub(r"\s+", " ", (e.get("description") or "").strip().lower()),
        )

    seen, deduped, removed = set(), [], 0
    for e in entries:
        k = _key(e)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        deduped.append(e)

    structured["experience"] = deduped
    return removed


_STRUCTURED_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_name": {"type": "string"},
        "current_role_title_from_summary": {"type": "string"},
        "stated_years_experience_from_summary": {"type": "number"},
        "skills": {"type": "array", "items": {"type": "string"}, "maxItems": 80},
        "experience": {
            "type": "array",
            "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "start_date_raw": {"type": "string"},
                    "end_date_raw": {"type": "string"},
                    "domain": {"type": "string"},
                    "technologies_used": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                },
                "required": ["title", "company", "start_date_raw", "end_date_raw"],
            },
        },
        "education": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "degree_level": {"type": "string"},
                    "field": {"type": "string"},
                    "institution": {"type": "string"},
                },
            },
        },
        "certifications": {"type": "array", "items": {"type": "string"}, "maxItems": 25},
        "projects": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "technologies_used": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                },
            },
        },
    },
    "required": ["candidate_name", "skills", "experience", "education", "certifications", "projects"],
}


def extract_structured_evidence_llm_legacy(resume_text):
    """Sends already-exact resume markdown to Qwen to extract structured JSON."""

    system_prompt = (
        "You are an expert AI Resume Extractor. Your task is to extract information from the provided resume "
        "and output it EXACTLY matching the following JSON schema. Do not include any conversational text outside the JSON.\n\n"
        "IMPORTANT RULES:\n"
        "- Extract EVERY item explicitly listed under each resume section (every skill, every experience entry, "
        "every project, every certification, every education entry) — do not stop early or summarize a subset.\n"
        "- Only place an item in \"projects\" if it appears under a section literally titled Projects (or similar). "
        "Do NOT create a project entry from an Experience/Internship bullet point, and do not duplicate the same "
        "content across two fields.\n"
        "- For each experience entry, copy start_date_raw and end_date_raw EXACTLY as written in the resume text "
        "(e.g. \"Oct 2023\", \"Mar 2022\", \"2019\"). Do NOT calculate a duration, a number of years, or a total — "
        "that is computed separately, outside your output.\n"
        "- If the resume says the role is ongoing — using ANY wording such as \"Present\", \"Current\", \"Till "
        "date\", \"Ongoing\", or similar — copy that exact word as end_date_raw. This is a literal copy of text "
        "that IS on the page, not a guess. Only use an empty string for end_date_raw in the rare case where the "
        "resume shows a start date but truly no end date and no ongoing-indicator word at all.\n"
        "- For each experience entry, also extract technologies_used: the specific tools/technologies/platforms "
        "listed for THAT role specifically — e.g. a line like \"Tools and Techniques Used – Python, Docker, "
        "Kubernetes\" appearing under that job. Keep this separate from the top-level skills list below.\n"
        "- For each project entry, extract technologies_used the same way — tools/technologies mentioned for "
        "that specific project. Keep this separate from both the top-level skills list and from any experience "
        "entry's technologies_used.\n"
        "- The top-level \"skills\" field is ONLY for items listed under a resume section literally titled "
        "Skills / Technical Skills / Skills and Technologies (or similar) — not items pulled from job or project "
        "descriptions, even if they'd fit there too.\n"
        "- An \"experience\" entry corresponds ONLY to a distinct job/role block under a Work Experience or "
        "Employment History section — one entry per unique (title, company, date range) combination actually "
        "printed on the resume. Do NOT create a separate experience entry for each category or grouping listed "
        "under Technical Skills / Skills (e.g. \"Data Engineering & Pipelines\", \"Programming & Frameworks\", "
        "\"Model Deployment\" are skill categories, never job roles).\n"
        "- Never repeat the same experience entry more than once. If you notice yourself about to write an "
        "entry with the same title, company, and dates as one you already wrote, stop — do not write it again.\n"
        "- If a field has no information in the resume, use an empty array or empty string — never invent content.\n\n"
        "CURRENT ROLE TITLE FROM SUMMARY (current_role_title_from_summary field):\n"
        "- If the resume has a Summary / Profile / About / Objective section that explicitly states the "
        "candidate's current or most recent job title in prose (e.g. \"Data Scientist with 5 years of "
        "experience in...\", \"Results-driven Senior Data Engineer specializing in...\"), copy JUST that "
        "title phrase exactly as written (e.g. \"Data Scientist\", \"Senior Data Engineer\").\n"
        "- This is separate from — and in addition to — the title you already extract for each entry in "
        "\"experience\" below. It exists because a resume's stated current title in prose doesn't always "
        "exactly match, or isn't always clearly the first/most-recent entry in, the dated experience list "
        "(unconventional ordering, a title only mentioned in prose, etc.).\n"
        "- Only extract this if the summary/profile text ACTUALLY states a title. Do NOT infer or guess a "
        "title from the candidate's skills, most recent employer, or general seniority level. If there is "
        "no summary/profile section, or it doesn't state a title, current_role_title_from_summary MUST be "
        "an empty string — that is correct and expected, not something to fill in.\n\n"
        "STATED YEARS OF EXPERIENCE FROM SUMMARY (stated_years_experience_from_summary field):\n"
        "- If the resume has a Summary / Profile / About section that explicitly states the candidate's "
        "total years of experience (e.g., \"5+ years of experience\", \"Over 8 years in data engineering\"), "
        "extract the numeric value (e.g., 5, 8).\n"
        "- Do NOT calculate or guess this number by looking at the job history. Only extract it if it is "
        "explicitly written in the summary prose. If it is not stated, output 0.\n\n"
        "SCHEMA:\n"
        "{\n"
        '  "candidate_name": "string (extracted from resume)",\n'
        '  "current_role_title_from_summary": "string — ONLY if the resume\'s summary/profile prose states '
        'a current/most-recent title; empty string otherwise",\n'
        '  "stated_years_experience_from_summary": "number — exact number stated in summary prose, or 0 if not stated",\n'
        '  "skills": ["string", "string"],\n'
        '  "experience": [\n'
        '    {\n'
        '      "title": "string",\n'
        '      "company": "string",\n'
        '      "start_date_raw": "string — copied exactly as written, e.g. \'Oct 2023\'",\n'
        '      "end_date_raw": "string — copied exactly as written, e.g. \'Present\' or \'Mar 2022\'",\n'
        '      "domain": "string (inferred domain/industry)",\n'
        '      "description": "string",\n'
        '      "technologies_used": ["string", "string"]\n'
        '    }\n'
        '  ],\n'
        '  "education": [\n'
        '    { "degree_level": "string", "field": "string", "institution": "string" }\n'
        '  ],\n'
        '  "certifications": ["string", "string"],\n'
        '  "projects": [\n'
        '    { "title": "string", "description": "string", "technologies_used": ["string", "string"] }\n'
        '  ]\n'
        "}"
    )

    response = ollama.chat(
        model='qwen2.5:3b-instruct',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Resume Text:\n{resume_text}"}
        ],
        format=_STRUCTURED_EVIDENCE_SCHEMA,
        options={
            'temperature': 0.0,
            'num_ctx': 8192,
        }
    )

    raw_output = response['message']['content']

    try:
        structured = json.loads(raw_output)
        removed = _dedupe_repeated_experience(structured)
        if removed:
            logger.warning(
                "Model repeated %d experience entr%s verbatim — collapsed via dedupe.",
                removed, "y" if removed == 1 else "ies"
            )
        return structured
    except json.JSONDecodeError:
        logger.warning("Model returned incomplete JSON — attempting salvage before discarding.")
        repaired = _attempt_json_repair(raw_output)
        if repaired is not None:
            removed = _dedupe_repeated_experience(repaired)
            note = f" ({removed} duplicate experience entr{'y' if removed == 1 else 'ies'} collapsed)" if removed else ""
            logger.warning("Recovered partial data from truncated model output via repair%s.", note)
            repaired.setdefault("_extraction_note", f"Recovered from truncated/malformed model output{note}.")
            return repaired
        print("\n[WARNING] Model failed to return a complete JSON object, and it could not be salvaged. Returning raw text.")
        return {"error": "Incomplete JSON", "raw_output": raw_output}


def _aggregate_skills(structured: dict) -> list:
    """Merges the resume's own Skills-section list with every technologies_used entry
    from experience roles and projects, deduped case-insensitively, preserving first-seen
    casing/order."""
    seen = set()
    aggregated = []

    def _add(item):
        if not item or not isinstance(item, str):
            return
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            aggregated.append(item)

    for skill in structured.get("skills", []) or []:
        _add(skill)
    for entry in structured.get("experience", []) or []:
        for tech in entry.get("technologies_used", []) or []:
            _add(tech)
    for proj in structured.get("projects", []) or []:
        for tech in proj.get("technologies_used", []) or []:
            _add(tech)

    # NEW: Extract skills from certifications formatted as "Label: skill, skill"
    for cert in structured.get("certifications", []) or []:
        if ":" in cert:
            _, content = cert.split(":", 1)
            for token in _split_dense_skill_line(content):
                _add(token)

    return aggregated


# --- Fabrication guards and deterministic recovery ---

def _normalize_for_guard(text: str) -> str:
    """Normalizes whitespace AND typographic quote/dash characters before comparison."""
    text = (text or "")
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
    return re.sub(r"\s+", " ", text.strip().lower())


def _text_supports_item(item: str, raw_text_norm: str) -> bool:
    """True if `item` is actually supported by the raw resume text."""
    item_norm = _normalize_for_guard(item)
    if not item_norm:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(item_norm) + r"(?![a-z0-9])"
    if re.search(pattern, raw_text_norm):
        return True

    words = item_norm.split()
    if len(words) > 1:
        return all(
            re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", raw_text_norm)
            for w in words
        )
    return False


def _guard_against_fabricated_skills(structured: dict, raw_text: str) -> list:
    """Drops any "skills" or "certifications" entry that doesn't actually appear
    anywhere in the candidate's raw resume text."""
    warnings = []
    raw_text_norm = _normalize_for_guard(raw_text)

    for field in ("skills", "certifications"):
        items = structured.get(field) or []
        if not items:
            continue

        kept, dropped = [], []
        for item in items:
            if _text_supports_item(item, raw_text_norm):
                kept.append(item)
            else:
                dropped.append(item)

        if dropped:
            warnings.append(
                f"{field}: discarded {len(dropped)} entr{'y' if len(dropped) == 1 else 'ies'} "
                f"not found anywhere in the raw resume text ({', '.join(dropped)}) — likely "
                f"model fabrication rather than something actually on the page. Verify manually."
            )
            structured[field] = kept

    return warnings


def _guard_against_fabricated_summary_title(structured: dict, raw_text: str) -> list:
    """Clears current_role_title_from_summary if it isn't actually supported by the
    raw resume text."""
    title = (structured.get("current_role_title_from_summary") or "").strip()
    if not title:
        return []

    raw_text_norm = _normalize_for_guard(raw_text)
    if _text_supports_item(title, raw_text_norm):
        return []

    structured["current_role_title_from_summary"] = ""
    return [
        f"current_role_title_from_summary: discarded {title!r} — not found anywhere in the raw "
        f"resume text — likely model fabrication rather than something actually stated in a "
        f"summary/profile section. Verify manually."
    ]


_LABELED_SKILL_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 &/\-]{1,40}):\s*(.+)$")
_SKILL_LABEL_HINT_RE = re.compile(
    r"skill|technolog|tool|stack|language|framework|pipeline|engineering|database", re.IGNORECASE
)
_MID_PREPOSITION_RE = re.compile(r"\b(for|with|in|of|using|on)\b", re.IGNORECASE)
_MAX_RECOVERED_TOKEN_WORDS = 3


def _split_dense_skill_line(content: str) -> list:
    """Splits a colon-labeled skill line's content into candidate atomic tokens."""
    candidates = []
    for part in re.split(r",|&", content):
        part = re.sub(r"^(?:and|or)\s+", "", part.strip(" ."), flags=re.IGNORECASE).strip(" .")
        if not part or _MID_PREPOSITION_RE.search(part) or len(part.split()) > _MAX_RECOVERED_TOKEN_WORDS:
            continue
        candidates.append(part)
    return candidates


def _recover_labeled_skill_lines(raw_text: str, structured: dict) -> list:
    """Returns the skill tokens recovered from raw_text that Stage 2 dropped."""
    existing_norm = {_normalize_for_guard(s) for s in (structured.get("skills") or [])}
    for entry in structured.get("experience", []) or []:
        existing_norm.update(_normalize_for_guard(t) for t in entry.get("technologies_used", []) or [])
    for proj in structured.get("projects", []) or []:
        existing_norm.update(_normalize_for_guard(t) for t in proj.get("technologies_used", []) or [])

    recovered = []
    for line in (raw_text or "").splitlines():
        m = _LABELED_SKILL_LINE_RE.match(line.strip())
        if not m or not _SKILL_LABEL_HINT_RE.search(m.group(1)):
            continue
        for token in _split_dense_skill_line(m.group(2)):
            token_norm = _normalize_for_guard(token)
            if not token_norm or token_norm in existing_norm:
                continue
            existing_norm.add(token_norm)
            recovered.append(token)

    if recovered:
        structured["skills"] = (structured.get("skills") or []) + recovered
    return recovered


_COMPANY_DATE_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9&.,'’\- ]{2,60}?)\s*\n\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})"
    r"\s*[-–—]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}|present|current|till date|ongoing)",
    re.IGNORECASE | re.MULTILINE,
)


def _check_experience_completeness(raw_text: str, structured: dict) -> list:
    """Returns warnings listing company names that look like a job header in
    raw_text but have no matching entry in structured['experience']."""
    text = raw_text or ""
    candidates = set()
    for m in _COMPANY_DATE_LINE_RE.finditer(text):
        name = re.sub(r"\s+", " ", m.group(1).strip())
        if len(name.split()) <= 6:
            candidates.add(name)

    extracted_companies_norm = {
        _normalize_for_guard(e.get("company") or "") for e in structured.get("experience", []) or []
    }

    missing = sorted(
        c for c in candidates
        if c and not any(_normalize_for_guard(c) in ec or ec in _normalize_for_guard(c)
                          for ec in extracted_companies_norm if ec)
    )

    if not missing:
        return []
    return [
        f"experience: found a company/date-range header in the raw resume text with no "
        f"matching entry in the extracted experience list ({', '.join(missing)}) — the "
        f"model may have dropped a job. Verify manually."
    ]


# --- Stage 2: deterministic, section-aware resume extraction (no LLM) ---

_RESUME_SECTIONS = {
    "skills": r"(?:technical |professional )?skills?(?:\s*(?:and|&)\s*(?:technologies|tools?))?|core competenc(?:y|ies)|technology stack",
    "experience": r"(?:professional |work |employment )?experience|work history|career history",
    "projects": r"(?:key |personal |academic )?projects?",
    "education": r"education(?:al background)?(?:\s*(?:and|&)\s*certifications?)?|academic qualifications?",
    "certifications": r"certifications?|licenses?",
    "summary": r"(?:professional )?(?:summary|profile)|objective|about me",
}
_SECTION_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(" + "|".join(_RESUME_SECTIONS.values()) + r")\s*:?\s*$", re.I)
_DATE_RANGE_RE = re.compile(
    r"\b(?P<start>(?:[A-Za-z]{3,9}\.?\s*\d{2,4}|\d{4}))\s*(?:-|\u2013|\u2014|to)\s*"
    r"(?P<end>(?:[A-Za-z]{3,9}\.?\s*\d{2,4}|\d{4}|present|current|ongoing|till date))\b", re.I)
_DEGREE_RE = re.compile(
    r"\b(?P<degree>b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?s\.?|m\.?s\.?|b\.?sc\.?|m\.?sc\.?|bca|mca|mba|pgdm|bachelor(?:'s)?|master(?:'s)?|ph\.?d)\b"
    r"(?:\s+(?:in|of)\s+(?P<field>[A-Za-z& /-]{2,80}))?", re.I)

# These patterns find technology mentions in prose.  Extraction preserves the
# exact resume wording; any alias/equivalence decisions belong to the matcher.
_RESUME_ALIASES = (
    ("Spark", r"\b(?:pyspark|apache spark|spark)\b"),
    ("Airflow", r"\b(?:apache )?airflow(?: dags?)?\b"),
    ("OCR", r"\bocr\b|optical character recognition"),
    ("dbt", r"\bdbt\b"), ("Dagster", r"\bdagster\b"),
    ("Kafka", r"\b(?:apache )?kafka\b"), ("SQL", r"\bsql\b"),
    ("Python", r"\bpython\b"), ("Docker", r"\bdocker\b"),
    ("Kubernetes", r"\bkubernetes\b|\bk8s\b"),
    ("FastAPI", r"\bfastapi\b"), ("Django", r"\bdjango\b"),
    ("Databricks", r"\b(?:azure )?databricks\b"),
)
_NON_SKILL_HEADINGS = re.compile(
    r"^(?:awards?|(?:other\s+)?achievements?(?:\s+and\s+personal\s+interests?)?|"
    r"personal\s+interests?|extra[- ]curricular|interests?|publications?|"
    r"languages?|volunteering|references?)$",
    re.I,
)
_GENERIC_SKILL_CATEGORY_LABELS = re.compile(
    r"^(?:cloud platforms?|data technologies|programming|databases?|infrastructure|"
    r"languages?|frameworks?|platforms?|tools?(?: and techniques)?|technologies|"
    r"certifications?|technical skills?|technical expertise|skills?(?: and technologies)?)$",
    re.I,
)
_PROJECT_TITLE_HINT = re.compile(r"\b(?:ai|automation|system|platform|generation|annotation|assessment|optim(?:isation|ization)|model|planner|pipeline|warehouse)\b", re.I)
_TECHNOLOGY_LABEL_RE = re.compile(
    r"^\s*(?:tech(?:nologies)?(?:\s+used)?|"
    r"tools?(?:\s*(?:and|&|/)\s*(?:technologies|techniques))?(?:\s+used)?|"
    r"technology stack|tech stack|frameworks?|languages?)\s*[:–—-]\s*(?P<items>.+)$",
    re.I,
)
_ROLE_WORD_RE = re.compile(
    r"\b(?:engineer|scientist|analyst|manager|developer|consultant|architect|"
    r"specialist|lead|intern|designer|administrator|coordinator|owner)\b", re.I
)
_ROLE_SENIORITY_WORDS = {
    "senior", "junior", "jr", "sr", "lead", "principal", "staff", "associate",
    "assistant", "intern", "trainee", "ii", "iii", "iv", "i",
}
_MONTH_NUMBER = {
    name: index for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}


def _clean_resume_markup(value: str) -> str:
    """Remove predictable DOCX/PDF-to-Markdown decoration without changing words."""
    value = re.sub(r"</?[^>]+>", "", value or "")
    # Markdown PDFs often place several bold category labels on one physical
    # line, e.g. "**Languages:** Python **Big Data:** Apache Spark".  Retain
    # each label boundary so the following skill parser never fuses columns.
    value = re.sub(r"\*\*([^*\n]{1,80}:)\*\*", r" § \1", value)
    value = re.sub(r"[*_`]+", "", value)
    # PDF extraction can join a role title to its following month, e.g.
    # "Business AnalystMay 2016". Restore that boundary before date parsing.
    value = re.sub(
        r"(?<=[a-z)])(?=(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{2,4}\b)",
        " ",
        value,
        flags=re.I,
    )
    # Keep ':' because it identifies an explicit skill category or tools list.
    return re.sub(r"\s+", " ", value).strip(" #-/\t")


def _section_for_heading(line: str) -> str | None:
    heading = _clean_resume_markup(line).casefold()
    if not heading or len(heading) > 80 or any(token in heading for token in (".", ",", ";")):
        return None
    for name, pattern in _RESUME_SECTIONS.items():
        # A section heading must be the whole short line.  Treating a sentence that
        # happens to contain “project” as a heading loses the surrounding experience.
        if re.fullmatch(pattern, heading, re.I):
            return name
    return None


def _section_with_inline_content(line: str) -> tuple[str | None, str]:
    """Recognise flattened PDF lines such as 'SKILLS AND TECHNOLOGIES Python, SQL'."""
    for name, pattern in _RESUME_SECTIONS.items():
        # Inside a project/role, "Technology Stack: ..." is a technology list,
        # not a new top-level Skills section.
        if name == "skills" and re.match(r"^technology stack\s*[:–—-]", line, re.I):
            continue
        match = re.match(rf"^(?:{pattern})(?:\s*[:–—-]?\s+)(?P<content>.+)$", line, re.I)
        if match:
            return name, match.group("content").strip()
    return None, ""


def _resume_section_lines(text: str) -> dict[str, list[str]]:
    sections = {name: [] for name in _RESUME_SECTIONS}
    current = "other"
    for raw in (text or "").replace("\r", "").splitlines():
        # Markdown/PDF conversion can stack markers ("- • Tools Used ...").
        # Remove every leading marker so the actual label remains detectable.
        line = _clean_resume_markup(re.sub(r"^\s*(?:(?:[-•])\s*)+", "", raw))
        if not line:
            continue
        section = _section_for_heading(line)
        if section:
            current = section
            continue
        else:
            inline_section, inline_content = _section_with_inline_content(line)
            if inline_section:
                current = inline_section
                sections[current].append(inline_content)
                continue
        if _NON_SKILL_HEADINGS.fullmatch(line):
            current = "other"
        elif (
            current == "skills"
            and _looks_like_project_title(line)
            and (
                line.isupper()
                or not (
                    sections[current]
                    and (
                        ":" in sections[current][-1]
                        or sections[current][-1].rstrip().endswith(",")
                    )
                )
            )
        ):
            current = "projects"
            sections[current].append(line)
        else:
            sections.setdefault(current, []).append(line)
    return sections


def _looks_like_project_title(line: str) -> bool:
    """Recognise an unlabelled project heading without classifying prose as skills."""
    words = line.split()
    return (
        3 <= len(words) <= 12
        and len(line) <= 100
        and line[:1].isupper()
        and "," not in line
        and ":" not in line
        and not line.rstrip().endswith(",")
        and not re.search(r"[.!?]$", line)
        and bool(_PROJECT_TITLE_HINT.search(line))
        and sum(word[:1].isupper() for word in words) >= max(2, len(words) // 2)
    )


def _coalesce_skill_lines(lines: list[str]) -> list[str]:
    """Join PDF-wrapped lines before splitting a candidate's own skill list."""
    merged = []
    continuation_prefixes = re.compile(
        r"^(?:analysis|airflow|engineering|face|generation|intelligence|learning|nifi)\b", re.I
    )
    for line in lines:
        previous = merged[-1] if merged else ""
        category_value = previous.partition(":")[2].strip() if ":" in previous else ""
        # A PDF can put words from the same phrase on separate visual columns:
        # "Deep" + "Learning," or "Azure" + "Document" + "Intelligence,".
        # Continue only an unfinished first/list item, never an entire sentence.
        continues_wrapped_phrase = (
            line[:1].isupper()
            and not _TECHNOLOGY_LABEL_RE.match(line)
            and not line.rstrip().endswith(":")
            and ":" not in line
            and not _looks_like_skill_summary_prose(line)
            and (
                category_value and "," not in category_value and len(line.split()) == 1
            )
        )
        if merged and (
            previous.endswith((",", "-", "/", "+", "&", ":"))
            or (previous.count("(") > previous.count(")") and ":" not in line)
            or (line[:1].islower() and ":" not in line)
            or continuation_prefixes.match(line)
            or continues_wrapped_phrase
        ):
            merged[-1] = f"{previous.rstrip('-')} {line}".strip()
        else:
            merged.append(line)
    return merged


def _split_skill_items(value: str) -> list[str]:
    """Split list punctuation while keeping commas inside a parenthesized skill."""
    # Text extractors occasionally duplicate an opening parenthesis at a wrapped
    # line. One opening parenthesis is enough to preserve the grouped tool list.
    value = re.sub(r"\(\s*\(", "(", value)
    items, buffer, depth = [], [], 0
    for index, character in enumerate(value):
        if character in "([{" :
            depth += 1
        elif character in ")] }".replace(" ", ""):
            depth = max(0, depth - 1)
        # A plus is a list delimiter only when it is visually used as one.
        # Do not split "5+ years" into a fake skill such as "AI Lead with 5".
        is_plus_delimiter = (
            character == "+"
            and index > 0
            and index + 1 < len(value)
            and value[index - 1].isspace()
            and value[index + 1].isspace()
        )
        if (character in ",;•|" or is_plus_delimiter) and depth == 0:
            items.append("".join(buffer))
            buffer = []
        else:
            buffer.append(character)
    items.append("".join(buffer))
    return [item.strip() for item in items]


def _parenthesized_skill_items(item: str) -> list[str]:
    """Expose exact items in a grouped list as well as the complete phrase.

    A resume can write ``Microsoft Azure (Data Factory, Databricks)``.  The
    complete phrase is useful evidence, but the independently listed Azure
    services must also be available for matching.  This works for any grouped
    comma list and does not rename or map the candidate's wording.
    """
    match = re.fullmatch(r"\s*([^()]{2,80}?)\s*\(([^()]{1,220})\)\s*", item)
    if not match or "," not in match.group(2):
        return []
    outer = match.group(1).strip(" .-–—")
    inner = [part.strip(" .-–—") for part in _split_skill_items(match.group(2))]
    return [value for value in [outer, *inner] if value]


def _looks_like_skill_summary_prose(text: str) -> bool:
    """Reject narrative text that happens to sit below a Skills heading."""
    normalized = re.sub(r"\s+", " ", text).strip()
    # A long comma-separated list is still a list; a long sentence is prose.
    if len(normalized) > 180 and normalized.count(",") < 2:
        return True
    return bool(re.search(
        r"\b(?:years?\s+of\s+experience|speciali[sz]ed|proven\s+ability|"
        r"responsible\s+for|developed|delivered|worked|designed|built|"
        r"implemented|ensured|managed|leading)\b",
        normalized,
        re.I,
    ))


def _explicit_technology_list_items(text: str) -> list[str]:
    """Read a comma-separated tool list introduced in normal resume prose.

    This is deliberately evidence-led: it accepts only lists explicitly
    introduced by phrases such as "using" or "leveraging", never every noun
    from a responsibility sentence.
    """
    matches = re.finditer(
        r"\b(?:using|leveraging|utilized|utilising|via)\s+([^.!?;]{1,260})",
        text,
        re.I,
    )
    items = []
    for match in matches:
        value = match.group(1).strip()
        # Stop before the explanatory part of a sentence.  For example,
        # "using LangGraph to ensure coherent trip generation" yields
        # LangGraph, not the prose after it.
        value = re.split(
            r"\b(?:to|for|while|ensuring|allowing|enabling|handling|covering|"
            r"reducing|improving)\b",
            value,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" ,")
        # A single named technology ("utilized Qdrant vector database") is
        # also evidence.  Reject only longer prose clauses without list syntax.
        if ("," not in value and not re.search(r"\s+and\s+", value, re.I)
                and len(value.split()) > 5):
            continue
        for item in _split_skill_items(re.sub(r"\s+and\s+", ", ", value, flags=re.I)):
            item = re.sub(r"^(?:and|with)\s+", "", item.strip(), flags=re.I)
            # These are explanatory continuations after a valid list, not tools.
            if re.match(r"^(?:to|for|while|reducing|ensuring|allowing|enabling|handling|covering)\b", item, re.I):
                continue
            # Project prose can name an input object ("a custom image dataset")
            # rather than a capability or tool.  Do not promote those objects to
            # skills merely because they follow "using".
            if (re.match(r"^(?:a|an|the)\b", item, re.I)
                    or re.search(r"\bdata(?:set)?s?$", item, re.I)):
                continue
            if 1 <= len(item.split()) <= 8 and len(item) <= 80:
                items.append(item)
    return items


def _summary_skill_items(text: str) -> list[str]:
    """Extract explicitly listed skills from a narrative summary sentence."""
    found = []
    # Parentheses in a summary commonly contain a concrete technology list.
    for group in re.findall(r"\(([^()]{1,300})\)", text):
        if "," in group:
            found.extend(_split_skill_items(group))
    # Keep only the list that follows an explicit skill-introducing phrase.
    for match in re.finditer(
        r"\b(?:skilled|proficient|experienced|expert)\s+in\s+([^.!?]{1,300})",
        text,
        re.I,
    ):
        value = re.split(r",?\s+with\s+(?:a|the)\b|\bthrough\b", match.group(1), maxsplit=1, flags=re.I)[0]
        found.extend(_split_skill_items(re.sub(r"\s+and\s+", ", ", value, flags=re.I)))
    # Preserve an explicit acronym-led capability before its example list,
    # e.g. "ETL/ELT pipelines (Azure Data Factory, ...)".
    for match in re.finditer(r"\b([A-Z]{2,}(?:/[A-Z]{2,})?\s+(?:pipelines?|workflows?))\b", text):
        found.append(match.group(1))
    return [item.strip(" .-–—") for item in found if item.strip(" .-–—")]


def _skills_with_evidence(lines: list[str], section: str) -> tuple[list[str], list[dict]]:
    """Extract candidate-written skill items, not only terms in a central catalog.

    The alias map only unifies known equivalents (for example, PySpark becomes
    Spark). It never decides whether a resume-specific item is a skill.
    """
    labels_with_source = []
    capability_list_continues = False
    # A "Tools and Techniques Used" list is commonly wrapped over several PDF
    # lines inside a project or an employment entry, not only in Skills.
    for line in _coalesce_skill_lines(lines):
        for segment in line.split("§"):
            text = segment.strip()
            if not text:
                continue
            technology_line = _TECHNOLOGY_LABEL_RE.match(text)
            left, separator, right = text.partition(":")
            is_labeled = bool(separator)
            # On page-two resumes, a "Certifications" region can contain grouped
        # capabilities such as "Data Engineering & Pipelines: Pandas, SQL".
        # Treat those labelled lists as skills, not as certificate names.
            grouped_capability_line = section == "Certifications" and is_labeled
            capability_continuation = (
            section == "Certifications"
            and capability_list_continues
            and not is_labeled
            and bool(re.fullmatch(r"[A-Za-z][A-Za-z .+/#&()'-]{0,80}", text))
            and len(text.split()) <= 6
            )
            narrative_skills_line = (
            section == "Skills"
            and not technology_line
            and not is_labeled
            and _looks_like_skill_summary_prose(text)
            )
            if ((section == "Skills" and not narrative_skills_line) or technology_line
                    or grouped_capability_line or capability_continuation):
                if technology_line:
                    values = [technology_line.group("items")]
                else:
                    # Category labels such as "Cloud Platforms" describe the
                    # list; they are not candidate skills.  Keep meaningful
                    # labels (e.g. "GenAI/RAG") as evidence when appropriate.
                    values = ([] if is_labeled and _GENERIC_SKILL_CATEGORY_LABELS.fullmatch(left.strip())
                              else ([left] if is_labeled else [])) + ([right] if is_labeled else [text])
                for value in values:
                    for item in _split_skill_items(value):
                        item = re.sub(r"\s+", " ", item).strip(" .-–—")
                        item = re.sub(r"^(?:and|including|with)\s+", "", item, flags=re.I)
                        if item.startswith("(") and item.count("(") > item.count(")"):
                            item = item.lstrip("(").strip()
                        if item.endswith(")") and item.count(")") > item.count("("):
                            item = item.rstrip(")").strip()
                        nested_items = _parenthesized_skill_items(item)
                        if (not item or len(item) > 80
                                or (len(item.split()) > 8 and not nested_items)
                                or _looks_like_skill_summary_prose(item)):
                            continue
                        if re.fullmatch(r"(?:skills?|tools?|technologies|frameworks?|languages|tech|misc|platforms?|core frameworks?|on)", item, re.I):
                            continue
                        # A long grouped cloud/platform list is useful only as
                        # its individual exact items; otherwise retain the
                        # candidate's original phrase as evidence too.
                        if len(item.split()) <= 8:
                            labels_with_source.append((item, text))
                        for nested_item in nested_items:
                            labels_with_source.append((nested_item, text))
                        # Preserve the complete candidate phrase, and also make
                        # a terminal named tool matchable ("indexing using
                        # FAISS" -> "FAISS") without relying on a dictionary.
                        terminal_tool = re.search(r"\busing\s+([A-Za-z][A-Za-z0-9+#._-]{1,})$", item, re.I)
                        if terminal_tool:
                            labels_with_source.append((terminal_tool.group(1), text))
            if section == "Certifications":
                # A visual skills column can become a labelled first line followed
                # by individual bullet lines after PDF extraction.
                capability_list_continues = grouped_capability_line or capability_continuation
            if section == "Summary":
                for item in _summary_skill_items(text):
                    if len(item.split()) <= 8:
                        labels_with_source.append((item, text))
            # Outside a formal Skills section we only use aliases when they appear
            # in ordinary prose. This catches “used PySpark” without treating every
            # noun in a responsibility sentence as a skill.
            if section in {"Experience", "Projects", "Summary"}:
                for item in _explicit_technology_list_items(text):
                    labels_with_source.append((item, text))
                for _canonical, pattern in _RESUME_ALIASES:
                    for mention in re.finditer(pattern, text, re.I):
                        labels_with_source.append((mention.group(0), text))
    seen, skills, evidence = set(), [], []
    for raw_label, source in labels_with_source:
        label = raw_label
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        skills.append(label)
        evidence.append({"skill": label, "source_section": section, "evidence": source[:300]})
    return skills, evidence


def _deterministic_experience(lines: list[str], fallback_title: str = "") -> list[dict]:
    entries = []
    for index, line in enumerate(lines):
        date = _DATE_RANGE_RE.search(line)
        if not date:
            continue
        next_role_index = next(
            (candidate_index for candidate_index in range(index + 1, len(lines))
             if _DATE_RANGE_RE.search(lines[candidate_index])),
            len(lines),
        )
        role_lines = lines[max(0, index - 3):next_role_index]
        context = " ".join(role_lines)
        nearby = lines[max(0, index - 2):index + 2]
        title_company = next((value for value in nearby if _ROLE_WORD_RE.search(value)), "")
        if not title_company:
            # Some PDF layouts put the date at the end of a large job block.  Use
            # the first explicit role/company header in that Experience section.
            title_company = next((value for value in lines if _ROLE_WORD_RE.search(value)), "")
        parts = re.split(r"\s*(?:\||@|,|\bat\b|—|–)\s*", title_company, maxsplit=1, flags=re.I)
        if len(parts) > 1:
            left, right = (part.strip() for part in parts)
            # Both ``Company | Role`` and ``Role | Company`` appear in real
            # resumes. Identify the side which is actually a title rather
            # than assuming one visual template.
            if _ROLE_WORD_RE.search(left) and not _ROLE_WORD_RE.search(right):
                title, company = left, right
            elif _ROLE_WORD_RE.search(right) and not _ROLE_WORD_RE.search(left):
                title, company = right, left
            else:
                company, title = left, right
        else:
            title, company = title_company.strip() or fallback_title, ""
            # Some templates omit a visual separator entirely, producing a
            # flattened header such as "Acme Technologies GenAI Engineer".
            # Keep the employer separately; only the demonstrated role suffix
            # may be used as job-title evidence downstream.
            title, flattened_company = split_flattened_company_title(title)
            if flattened_company:
                company = flattened_company
        # Some templates put the dates first, then the employer on the next
        # line, and the role only in the resume headline.  Keep the headline as
        # the role but still store the employer for that entry.
        if not company:
            company = next((value.strip(" -–—,()") for value in lines[index + 1:min(next_role_index, index + 4)]
                            if re.search(r"\b(?:inc(?:orporated)?|ltd|llc|corp(?:oration)?|"
                                         r"technologies|solutions|systems|company|pvt)\b", value, re.I)), "")
        title = _DATE_RANGE_RE.sub("", title).strip(" -–—,")
        # PDF text occasionally duplicates a closing parenthesis beside a
        # date. Preserve legitimate parenthetical titles while removing only
        # surplus trailing punctuation.
        while title.endswith(")") and title.count(")") > title.count("("):
            title = title[:-1].rstrip()
        # A date visually placed beside a summary must not become a fake role.
        if len(title) > 100 or re.search(r"\byears?\s+of\s+experience\b", title, re.I):
            continue
        # Keep physical lines separate: an explicit "Tools and Techniques Used"
        # label may appear after the role date, and matching it requires the line
        # to begin with that label.
        tech, _ = _skills_with_evidence(role_lines, "Experience")
        entries.append({"title": title, "company": company, "start_date_raw": date.group("start"),
                        "end_date_raw": date.group("end"), "domain": "", "technologies_used": tech})
    return _dedupe_experience_entries(entries)


def _dedupe_experience_entries(entries: list[dict]) -> list[dict]:
    seen, output = set(), []
    for entry in entries:
        key = tuple((entry.get(field) or "").casefold() for field in ("title", "company", "start_date_raw", "end_date_raw"))
        if key not in seen:
            seen.add(key)
            output.append(entry)
    return output


def _columnar_experience_fallback(lines: list[str]) -> list[dict]:
    """Recover role/date columns when PDF reading order separates them.

    This activates only when regular section-based extraction produced nothing.
    It pairs short role/company headers with the first matching employment date
    ranges, deliberately ignoring later academic dates.
    """
    headers = []
    for index, line in enumerate(lines):
        if len(line) > 110 or not _ROLE_WORD_RE.search(line) or "|" not in line:
            continue
        left, right = (part.strip() for part in line.split("|", 1))
        if _ROLE_WORD_RE.search(left):
            title, company = left, right
        elif _ROLE_WORD_RE.search(right):
            title, company = right, left
        else:
            continue
        headers.append((index, title, company))
    dates = [(index, match) for index, line in enumerate(lines)
             if (match := _DATE_RANGE_RE.search(line))]
    if not headers or len(dates) < len(headers):
        return []
    entries = []
    for (header_index, title, company), (_date_index, date) in zip(headers, dates):
        # Keep local evidence near the role header plus its corresponding date.
        start = header_index
        end = min(len(lines), header_index + 35)
        tech, _ = _skills_with_evidence(lines[start:end], "Experience")
        entries.append({"title": title, "company": company,
                        "start_date_raw": date.group("start"), "end_date_raw": date.group("end"),
                        "domain": "", "technologies_used": tech})
    return _dedupe_experience_entries(entries)


def _labelled_skill_lines(lines: list[str]) -> list[str]:
    """Find compact 'category: skill, skill' lines anywhere in a shuffled PDF."""
    selected = []
    for line in lines:
        left, separator, right = line.partition(":")
        is_labelled_list = (
            separator and 1 <= len(left.split()) <= 5 and len(line) <= 400
            and right.count(",") >= 1
            and not re.search(r"\b(?:built|orchestrated|engineered|integrated|"
                              r"process|delivery|architecture|modernization)\b", left, re.I)
            and not re.match(r"^(?:description|responsibilities|contact|address|phone|email)$", left, re.I)
        )
        if is_labelled_list:
            selected.append(line)
            continue

        # Preserve a wrapped row in a visual skills table. The following line
        # has no category label of its own, so it belongs to the preceding
        # category until the next labelled row begins.
        previous = selected[-1] if selected else ""
        if previous and not separator and (
            previous.count("(") > previous.count(")")
            or previous.rstrip().endswith((",", "-", "/", "+", "&", ":"))
            or re.match(r"^(?:analysis|airflow|engineering|face|generation|intelligence|learning|nifi)\b", line, re.I)
        ):
            selected.append(line)
    return selected


def _deterministic_projects(lines: list[str]) -> list[dict]:
    projects = []
    for index, line in enumerate(lines):
        normalized_title = re.sub(r"^\s*(?:project\s*)?\d+\s*[:-]+\s*", "", line, flags=re.I)
        if len(normalized_title) > 100 or _DATE_RANGE_RE.search(normalized_title) or not _looks_like_project_title(normalized_title):
            continue
        title = re.sub(r"^[-•#\s]+", "", normalized_title).strip()
        if not title or title.lower().startswith(("tools", "technologies", "description")):
            continue
        # A project can span a whole page.  Retain its evidence through the
        # next recognisable project heading instead of inspecting only its
        # first few visual lines.
        next_project_index = next(
            (candidate_index for candidate_index in range(index + 1, len(lines))
             if _looks_like_project_title(re.sub(r"^\s*(?:project\s*)?\d+\s*[:-]+\s*", "", lines[candidate_index], flags=re.I))),
            len(lines),
        )
        project_lines = lines[index:next_project_index]
        context = " ".join(project_lines)
        tech, _ = _skills_with_evidence(project_lines, "Projects")
        project = {"title": title[:120], "technologies_used": tech}
        if tech:
            project["description"] = context[:500]
        projects.append(project)
    return projects


def _headline_title(lines: list[str]) -> str:
    """Read an explicit title near the top; never infer a title from skills."""
    for line in lines[:15]:
        candidate = _clean_resume_markup(line)
        if ("@" not in candidate and not re.search(r"\d", candidate)
                and re.fullmatch(r"[A-Za-z][A-Za-z .&/()'’-]{2,80}", candidate)
                and re.search(r"\b(?:engineer|scientist|analyst|manager|developer|consultant|architect|designer|lead|intern)\b", candidate, re.I)):
            return candidate
    return ""


def _experience_date_key(raw_date: str) -> tuple[int, int]:
    """Sortable end-date key used only to find the latest extracted job."""
    value = (raw_date or "").strip().casefold()
    if re.fullmatch(r"(?:present|current|ongoing|till date)", value):
        return (9999, 12)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    if not year_match:
        return (0, 0)
    month_match = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep(?:t)?|oct|nov|dec)[a-z]*\.?", value)
    month = _MONTH_NUMBER.get(month_match.group(1)[:3], 12) if month_match else 12
    return (int(year_match.group(1)), month)


def _latest_experience_role(experience: list[dict]) -> tuple[str, dict | None]:
    valid = [entry for entry in experience if _ROLE_WORD_RE.search((entry.get("title") or ""))]
    if not valid:
        return "", None
    latest = max(valid, key=lambda entry: _experience_date_key(entry.get("end_date_raw") or ""))
    return (latest.get("title") or "").strip(), latest


def _resolve_current_role(latest_title: str, headline_title: str) -> tuple[str, str]:
    """Use employment history for the role used in screening.

    A profile/summary often describes an aspiration, a broader professional
    identity, or several roles.  It must not override the candidate's most
    recent dated employment title.  A clearly presented headline title is a
    fallback only when no work-history title was extracted.
    """
    if latest_title:
        return latest_title, "latest_experience"
    if headline_title:
        return headline_title, "resume_headline_fallback"
    return "", "no_current_or_latest_role_found"


def _extract_certifications(lines: list[str]) -> list[str]:
    """Avoid storing ordinary skills/project prose as certifications."""
    markers = re.compile(r"\b(?:cert(?:ificate|ification|ified)?|course|credential|license|professional|fundamentals)\b", re.I)
    return [line for line in lines if len(line) <= 180 and markers.search(line)]


def extract_structured_evidence(resume_text: str) -> dict:
    """Deterministically structure a resume; this function makes no model calls."""
    sections = _resume_section_lines(resume_text)
    all_lines = [_clean_resume_markup(line) for line in (resume_text or "").splitlines() if line.strip()]
    summary_candidates = [
        line for line in all_lines
        if re.search(r"\b\d+(?:\.\d+)?\+?\s+years?\b", line, re.I)
        and re.search(r"\b(?:experience|architecting|designing|working|building)\b", line, re.I)
    ]
    summary_lines = list(dict.fromkeys(sections["summary"] + summary_candidates))
    headline_title = _headline_title(all_lines)
    skills, evidence = _skills_with_evidence(sections["skills"], "Skills")
    experience = _deterministic_experience(sections["experience"], headline_title)
    projects = _deterministic_projects(sections["projects"])
    # Some multi-column PDFs render headings after their content.  Recover
    # structured lists/roles from stable local patterns instead of page order.
    labelled_lines = _labelled_skill_lines(all_lines)
    found, found_evidence = _skills_with_evidence(labelled_lines, "Skills")
    skills.extend(found)
    evidence.extend(found_evidence)
    if not experience:
        experience = _columnar_experience_fallback(all_lines)
    if not projects:
        projects = _deterministic_projects(all_lines)
    for section_name in ("summary", "experience", "projects", "certifications"):
        source_lines = summary_lines if section_name == "summary" else sections[section_name]
        found, found_evidence = _skills_with_evidence(source_lines, section_name.title())
        skills.extend(found)
        evidence.extend(found_evidence)
    # Capture an explicit Tech/Tools/Technology Stack label even when a PDF's
    # layout prevented its parent heading from being classified correctly.
    found, found_evidence = _skills_with_evidence(all_lines, "Resume")
    skills.extend(found)
    evidence.extend(found_evidence)
    skills = list(dict.fromkeys(skills))
    education = []
    for line in all_lines:
        for match in _DEGREE_RE.finditer(line):
            education.append({"degree_level": match.group("degree"), "field": (match.group("field") or "").strip(), "institution": ""})
    certifications = _extract_certifications(sections["certifications"] + sections["education"])
    candidate_name = next((line for line in all_lines[:25]
                           if re.fullmatch(r"[A-Z][A-Z .'-]{2,60}", line)
                           and not _section_for_heading(line)
                           and not re.search(r"\b(?:skills?|expertise|experience|education|projects?|"
                                             r"summary|profile|certifications?)\b", line, re.I)), "")
    if not candidate_name:
        candidate_name = next((line for line in all_lines[:25] if 1 < len(line.split()) <= 5 and "@" not in line and not re.search(r"\d", line) and not _section_for_heading(line)), "")
    summary_text = " ".join(summary_lines)
    years_match = re.search(r"\b(\d{1,2}(?:\.\d+)?)\s*\+?\s*years?\b", summary_text, re.I)
    latest_experience_role_title, _ = _latest_experience_role(experience)
    resolved_role_title, role_resolution = _resolve_current_role(
        latest_experience_role_title, headline_title
    )
    return {"candidate_name": candidate_name,
            # Kept under this established persistence key for backward compatibility.
            # It contains the current/latest employment role, never a summary claim.
            "current_role_title_from_summary": resolved_role_title,
            "latest_experience_role_title": latest_experience_role_title,
            "current_role_title_resolution": role_resolution,
            "stated_years_experience_from_summary": float(years_match.group(1)) if years_match else 0,
            "skills": skills, "experience": experience, "education": education,
            "certifications": certifications, "projects": projects, "skill_evidence": evidence}


def ingest_resume(file_path: str) -> dict:
    """
    Full resume ingestion:
      Stage 1 — extract exact text (pymupdf4llm / docx-to-markdown, OCR fallback)
      Stage 2 — structure using deterministic section and regex rules (no LLM)
      Stage 3 — compute total_years_experience deterministically from the extracted
                date ranges (experience.py), via range union
    then attach candidate_id/document_id/file_name/extraction metadata.
    """
    file_name = os.path.basename(file_path)
    raw_text, warnings, method = extract_text(file_path)

    _debug_dump_path = os.path.join("/tmp", f"raw_text_debug__{file_name}.txt")
    with open(_debug_dump_path, "w", encoding="utf-8") as _f:
        _f.write(f"[extraction_method={method}]\n[char_count={len(raw_text)}]\n\n{raw_text}")
    print(f"[DEBUG] Stage 1 raw_text ({len(raw_text)} chars, method={method}) written to {_debug_dump_path}")

    structured = extract_structured_evidence(raw_text)

    total_years, experience_warnings = compute_total_years(structured.get("experience", []))
    
    # NEW: Check if the summary explicitly stated a higher number of years
    stated_years = structured.get("stated_years_experience_from_summary", 0)
    
    if isinstance(stated_years, (int, float)) and stated_years > total_years:
        structured["total_years_experience"] = float(stated_years)
        warnings.append(f"experience: Used stated experience from summary ({stated_years}y) because it exceeded calculated experience ({total_years}y).")
    else:
        structured["total_years_experience"] = total_years
        
    structured["skills_all_sources"] = _aggregate_skills(structured)
    warnings = warnings + experience_warnings

    for w in warnings:
        logger.warning("[%s] %s", file_name, w)

    structured["candidate_id"] = new_id()
    structured["document_id"] = new_id()
    structured["file_name"] = file_name
    structured["extraction_method"] = method
    structured["extraction_warnings"] = warnings
    structured["uploaded_at"] = now_iso()
    return structured


if __name__ == "__main__":
    sample_resume = os.path.join(DATA_DIR, "Naukri_Ulka[5y_0m].pdf")

    if os.path.exists(sample_resume):
        print("Ingesting resume...")
        structured_data = ingest_resume(sample_resume)
        print(json.dumps(structured_data, indent=2))
    else:
        print(f"Please ensure your test resume is placed in: {DATA_DIR}")
