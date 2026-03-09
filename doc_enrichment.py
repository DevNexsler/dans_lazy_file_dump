"""LLM-based document enrichment for the indexing pipeline.

Sends a representative sample of each document's text to a generative LLM
(via configurable LLM provider) and parses its structured JSON response into
consistent metadata fields stored in LanceDB.

For documents longer than max_input_chars, uses head+tail sampling (first
half + last half) so that conclusions, summaries, and late-document facts
are captured alongside the opening context.

All fields are returned as strings (comma-separated for lists, JSON array
for key_facts) for consistent querying and filtering.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from providers.llm import LLMGenerator
    from taxonomy_store import TaxonomyStore

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Extract metadata from this document. Respond with ONLY valid JSON, no other text.

{{
  "summary": "2-3 sentence summary of the document's purpose and key content",
  "doc_type": ["type1", "type2"],
  "entities_people": ["full names of people mentioned"],
  "entities_places": ["addresses, cities, locations"],
  "entities_orgs": ["company and organization names"],
  "entities_dates": ["YYYY-MM-DD format dates mentioned"],
  "topics": ["5-10 high-level topics"],
  "keywords": ["10-20 specific terms and phrases"],
  "key_facts": ["most important facts, conclusions, or action items"],
  "suggested_tags": ["classification tags for this document"],
  "suggested_folder": "best folder path for filing this document",
  "importance": 0.5
}}

For "importance": rate the document's overall importance/usefulness on a 0.0-1.0 scale:
- 1.0 = critical reference, frequently needed, high-value knowledge
- 0.7-0.9 = important, actionable, or broadly useful
- 0.4-0.6 = average utility, general notes or routine content
- 0.1-0.3 = low importance, ephemeral, or narrowly relevant
- 0.0 = trivial, outdated, or noise
{taxonomy_block}
Document title: {title}
Document type: {source_type}

Document text:
{text}"""

_TAXONOMY_INSTRUCTION = """
For "suggested_tags" and "suggested_folder": use the taxonomy below.
Pick the most relevant tags from Available Tags (you may also add new ones).
Pick the single best matching folder path from Available Folders (use the exact path).
"""

# Raw keys the LLM prompt asks for (unprefixed)
_ENRICHMENT_KEYS_RAW = (
    "summary",
    "doc_type",
    "entities_people",
    "entities_places",
    "entities_orgs",
    "entities_dates",
    "topics",
    "keywords",
    "key_facts",
    "suggested_tags",
    "suggested_folder",
    "importance",
)

# Prefixed field names stored in LanceDB metadata (prevent collision with frontmatter)
ENRICHMENT_FIELDS = tuple(f"enr_{k}" for k in _ENRICHMENT_KEYS_RAW)


def empty_enrichment() -> dict[str, str]:
    """Return a dict with all enrichment fields set to empty strings."""
    return {f: "" for f in ENRICHMENT_FIELDS}


def failed_enrichment(reason: str) -> dict[str, str]:
    """Return an enrichment dict that signals failure with a reason.

    The caller should check for ``_enrichment_failed`` and remove it
    before storing in LanceDB.
    """
    result = empty_enrichment()
    result["_enrichment_failed"] = reason
    return result


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output, stripping markdown fences.

    Handles common LLM quirks:
      - Markdown ```json fences
      - <think>...</think> tags (Qwen3 reasoning)
      - Trailing text after valid JSON ("Extra data" errors)
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Handle Qwen3 thinking tags — strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Use JSONDecoder to parse only the first JSON object, ignoring trailing text
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(cleaned)
        return obj
    except json.JSONDecodeError:
        pass

    # Attempt to salvage truncated JSON (e.g. token limit cut off mid-value).
    # Progressively strip trailing incomplete tokens and try to close the object.
    salvaged = _salvage_truncated_json(cleaned)
    if salvaged is not None:
        return salvaged

    # Nothing worked — raise a clear error
    return json.loads(cleaned)


def _salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Try to recover a partial JSON object truncated by token limits.

    Strips trailing incomplete values and attempts to close open arrays/objects.
    Returns the parsed dict or None if recovery fails.
    """
    s = text.rstrip()
    # Try closing open structures, stripping up to 200 chars from the tail
    for trim in range(0, min(200, len(s))):
        candidate = s if trim == 0 else s[:-trim]
        # Remove trailing comma
        candidate = candidate.rstrip().rstrip(",").rstrip()
        # Count open/close brackets to figure out what needs closing
        closers = ""
        for ch in candidate:
            if ch in ('{', '['):
                closers = ('}' if ch == '{' else ']') + closers
            elif ch in ('}', ']') and closers and closers[0] == ch:
                closers = closers[1:]  # doesn't match LIFO but close enough
        # Close any remaining open brackets
        # Actually, rebuild closers by scanning properly
        stack = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append('}' if ch == '{' else ']')
            elif ch in ('}', ']') and stack:
                stack.pop()
        # If we're inside a string, close it first
        if in_string:
            candidate += '"'
        closers = "".join(reversed(stack))
        attempt = candidate + closers
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                logger.info("Salvaged truncated JSON (%d chars trimmed)", trim)
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_list(value: Any) -> str:
    """Convert a list (or string) to a comma-separated string."""
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value else ""


def _normalize_enrichment(raw: dict[str, Any]) -> dict[str, str]:
    """Normalize raw LLM JSON into consistent prefixed string fields."""
    result: dict[str, str] = {}
    for raw_key, enr_key in zip(_ENRICHMENT_KEYS_RAW, ENRICHMENT_FIELDS):
        value = raw.get(raw_key)
        if value is None:
            result[enr_key] = ""
        elif raw_key == "importance":
            # Normalize to a clamped float string
            try:
                imp = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                imp = 0.5
            result[enr_key] = str(imp)
        elif raw_key in ("summary", "suggested_folder"):
            result[enr_key] = str(value).strip()
        elif raw_key == "key_facts":
            if isinstance(value, list):
                result[enr_key] = json.dumps(
                    [str(v).strip() for v in value if str(v).strip()]
                )
            elif isinstance(value, str):
                result[enr_key] = value.strip()
            else:
                result[enr_key] = ""
        else:
            result[enr_key] = _normalize_list(value)
    return result


def enrich_document(
    text: str,
    title: str,
    source_type: str,
    generator: "LLMGenerator",
    max_input_chars: int = 4000,
    max_output_tokens: int = 512,
    taxonomy_store: "TaxonomyStore | None" = None,
) -> dict[str, str]:
    """Extract structured metadata from document text using an LLM.

    Returns a dict with all ENRICHMENT_FIELDS populated (or empty strings
    on failure).  Never raises — logs warnings on parse errors.
    """
    if not text or not text.strip():
        logger.debug("Skipping enrichment for empty document: %s", title)
        return empty_enrichment()

    if len(text) <= max_input_chars:
        truncated = text
    else:
        # Head + tail sampling: capture both opening context and late-document
        # conclusions/facts that a simple head truncation would miss.
        half = max_input_chars // 2
        truncated = text[:half] + "\n\n[...]\n\n" + text[-half:]

    # Build taxonomy context block for the prompt
    taxonomy_block = ""
    if taxonomy_store is not None:
        try:
            raw_block = taxonomy_store.format_for_prompt()
            if raw_block:
                taxonomy_block = f"\n{_TAXONOMY_INSTRUCTION}\n{raw_block}\n"
        except Exception as exc:
            logger.warning("Failed to load taxonomy for prompt: %s", exc)

    prompt = _PROMPT_TEMPLATE.format(
        title=title,
        source_type=source_type,
        text=truncated,
        taxonomy_block=taxonomy_block,
    )

    try:
        raw_response = generator.generate(prompt, max_tokens=max_output_tokens)
        logger.debug("LLM enrichment raw response for '%s': %s", title, raw_response[:200])

        parsed = _extract_json(raw_response)
        enrichment = _normalize_enrichment(parsed)

        if not enrichment.get("enr_summary"):
            logger.warning(
                "LLM returned empty summary for '%s'. Raw response: %s",
                title, raw_response[:300],
            )

        # Increment usage_count for matched taxonomy entries
        if taxonomy_store is not None:
            try:
                for tag in (enrichment.get("enr_suggested_tags") or "").split(","):
                    tag = tag.strip()
                    if tag:
                        taxonomy_store.increment_usage(f"tag:{tag}")
                folder = (enrichment.get("enr_suggested_folder") or "").strip()
                if folder:
                    taxonomy_store.increment_usage(f"folder:{folder}")
            except Exception as exc:
                logger.warning("Failed to increment taxonomy usage: %s", exc)

        logger.info(
            "Enriched '%s': doc_type=%s, topics=%s",
            title,
            enrichment.get("enr_doc_type", ""),
            enrichment.get("enr_topics", "")[:80],
        )
        return enrichment

    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse LLM JSON for '%s': %s. Response: %s",
            title, exc, raw_response[:300] if "raw_response" in dir() else "N/A",
        )
        return failed_enrichment(f"json_parse_error: {exc}")
    except Exception as exc:
        logger.error(
            "LLM enrichment failed for '%s': %s: %s", title, type(exc).__name__, exc,
        )
        return failed_enrichment(f"{type(exc).__name__}: {exc}")
