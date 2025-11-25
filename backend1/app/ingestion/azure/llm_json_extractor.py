# llm_json_extractor.py
import json
import re
from typing import Optional

def _strip_code_fences(text: str) -> str:
    """
    Remove triple-backtick fences (optionally with json hint) and leading/trailing whitespace.
    """
    if text is None:
        return ""
    # remove ```json or ``` and closing ```
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```", "", text)
    return text.strip()

def _attempt_fix_trailing_commas(s: str) -> str:
    """
    Simple fixes: remove trailing commas before } or ].
    This won't fix all invalid JSON, but handles a common LLM mistake.
    """
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s

def extract_json_str(text: str) -> Optional[str]:
    """
    Attempt to extract the first JSON object string from `text`.
    Returns the JSON string if found and reparsable (or reparsed with simple fixes),
    otherwise returns None.
    Approach:
      1. Strip code fences / markdown wrappers.
      2. Find the first '{' and perform bracket-matching to the corresponding closing '}'.
      3. Validate via json.loads; if invalid, attempt simple fixes and retry.
      4. Fallback: non-greedy regex search for {...} then same validation/repair.
    """
    if not text:
        return None

    # 1) Strip obvious code fences and surrounding whitespace
    t = _strip_code_fences(text)

    # Also remove leading/trailing lines like "Response:" or triple backtick with label
    # (keep it simple — main goal is to reach the JSON starting point)
    # Find first '{'
    first_open = t.find("{")
    if first_open == -1:
        # fallback to regex search (slightly more permissive)
        m = re.search(r"\{", t)
        if not m:
            return None
        first_open = m.start()

    # 2) Bracket-matching from first_open
    depth = 0
    start = None
    for i, ch in enumerate(t[first_open:], start=first_open):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = t[start:i + 1]
                # Try parsing candidate
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # attempt simple repairs
                    fixed = _attempt_fix_trailing_commas(candidate)
                    try:
                        json.loads(fixed)
                        return fixed
                    except Exception:
                        # continue searching (maybe nested JSON earlier failed)
                        pass
                # If candidate failed, stop bracket-match loop and fallback to regex below
                break

    # 3) Fallback: use a non-greedy regex to find the first {...}
    # Note: this may be slower but is a sensible fallback
    regex = re.compile(r"(\{(?:[^{}]|(?R))*\})", re.DOTALL)
    try:
        m = regex.search(t)
    except re.error:
        # Python's re doesn't support (?R) in standard library; fallback to simpler regex
        m = re.search(r"\{(?:.|\s)*?\}", t)

    if m:
        s = m.group(0)
        try:
            json.loads(s)
            return s
        except Exception:
            fixed = _attempt_fix_trailing_commas(s)
            try:
                json.loads(fixed)
                return fixed
            except Exception:
                return None

    return None


def extract_json(text: str) -> Optional[dict]:
    """
    Convenience wrapper: extract JSON string then parse to dict.
    Returns parsed dict or None.
    """
    s = extract_json_str(text)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # attempt fix and parse again
        try:
            s2 = _attempt_fix_trailing_commas(s)
            return json.loads(s2)
        except Exception:
            return None
