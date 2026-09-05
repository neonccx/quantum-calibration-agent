"""Reproducible, modality-aware QCalEval support.

The official benchmark mixes programmatic scoring with LLM judging.  This
module deliberately computes only the reproducible subset and labels the
result accordingly; it never manufactures an "official" overall score.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any


QUESTION_NAMES = {
    "Q1": "technical_description",
    "Q2": "experimental_conclusion",
    "Q3": "experimental_significance",
    "Q4": "fit_reliability",
    "Q5": "parameter_extraction",
    "Q6": "calibration_diagnosis",
}

KNOWN_STATUSES = {
    "SUCCESS", "NO_SIGNAL", "BEATING", "NO_DETUNING", "SAMPLING_TOO_COARSE",
    "TOO_FEW_OSC", "TOO_MANY_OSC", "WINDOW_TOO_SHORT", "DAMPED", "FIT_POOR",
    "RANGE_TOO_NARROW", "AMP_TOO_HIGH", "NO_EXCITATION", "NO_RES_RESPONSE",
    "HIGH_POWER", "OPTIMAL_NOT_CENTERED", "LARGE_ERROR", "MODERATE_ERROR",
    "FIT_FAILED", "MULTIPLE_PEAKS", "INVERTED", "INCOMPLETE", "NO_TRANSITION",
    "NEGATIVE_OFFSET", "POSITIVE_OFFSET", "EVENT", "NO_EVENT", "NO_COHERENCE",
    "STABLE", "TELEGRAPHIC", "RANDOM_WALK", "ASYMMETRIC", "UNDERSAMPLED",
    "NOT_TUNABLE", "OFF_RESONANCE", "LOW_CONTRAST", "DETUNED", "NO_GATE",
    "MISCALIBRATED", "ABERRATED", "CORRECTED",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    unsigned = dict(snapshot)
    unsigned.pop("snapshot_sha256", None)
    return sha256_bytes(canonical_json(unsigned))


def validate_snapshot(snapshot_path: Path) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != "qmagent.qcaleval.snapshot.v1":
        raise ValueError("unsupported QCalEval snapshot schema")
    recorded = snapshot.get("snapshot_sha256")
    actual = snapshot_digest(snapshot)
    if not recorded or recorded != actual:
        raise ValueError(f"snapshot SHA-256 mismatch: recorded={recorded!r}, actual={actual}")
    if not snapshot.get("source", {}).get("revision"):
        raise ValueError("snapshot source revision is required")
    entries = snapshot.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("snapshot must contain at least one entry")
    root = snapshot_path.parent.resolve()
    for entry in entries:
        if not entry.get("id") or not entry.get("experiment_type"):
            raise ValueError("every entry requires id and experiment_type")
        for label in QUESTION_NAMES:
            if label not in entry.get("prompts", {}):
                raise ValueError(f"entry {entry.get('id')} is missing prompt {label}")
        for image in entry.get("images", []):
            image_path = (root / image["path"]).resolve()
            if root not in image_path.parents:
                raise ValueError("snapshot image path escapes snapshot directory")
            if sha256_file(image_path) != image.get("sha256"):
                raise ValueError(f"image SHA-256 mismatch: {image['path']}")
    return snapshot


def image_content_blocks(snapshot_path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    root = snapshot_path.parent.resolve()
    blocks = []
    for image in entry.get("images", []):
        image_path = (root / image["path"]).resolve()
        mime = image.get("mime_type", "image/png")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return blocks


def parse_json_answer(text: str | None) -> Any:
    if not text:
        return None
    value = text.strip()
    candidates = [value]
    candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)```", value, re.IGNORECASE))
    object_match = re.search(r"(\{[\s\S]*\})", value)
    array_match = re.search(r"(\[[\s\S]*\])", value)
    candidates.extend(match.group(1) for match in (object_match, array_match) if match)
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _extract_labeled(text: str | None, label: str, known: list[str]) -> str | None:
    if not text:
        return None
    match = re.search(rf"{re.escape(label)}:\s*\*?\*?(.+?)(?:\*\*)?$", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip().rstrip(".").replace("**", "").strip()
    for candidate in known:
        if candidate.lower() in value.lower():
            return candidate
    return value


def extract_classification(text: str | None) -> str | None:
    return _extract_labeled(text, "Classification", [
        "Expected behavior", "Anomalous behavior", "Suboptimal parameters", "Apparatus issue",
    ])


def extract_assessment(text: str | None) -> str | None:
    return _extract_labeled(text, "Assessment", ["Unreliable", "No fit", "Reliable"])


def extract_status(text: str | None) -> str | None:
    if not text:
        return None
    for label in ("Status", "Classification"):
        match = re.search(rf"\b{label}:\s*\*{{0,2}}\s*([A-Z][A-Za-z_]+)", text)
        if match and match.group(1).upper() in KNOWN_STATUSES:
            return match.group(1).upper()
    first = text.strip().splitlines()[0].strip().replace("**", "").upper()
    if first in KNOWN_STATUSES:
        return first
    upper = text.upper()
    for status in sorted(KNOWN_STATUSES, key=len, reverse=True):
        if re.search(rf"\b{status}\b", upper):
            return status
    return None


def _enum_score(actual: Any, expected: Any) -> float:
    if actual is None or expected is None:
        return 0.0
    return float(str(actual).strip().lower() == str(expected).strip().lower())


def _array_f1(actual: Any, expected: Any, tolerance: float) -> float:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return 0.0
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    used: set[int] = set()
    hits = 0
    for value in actual:
        for index, target in enumerate(expected):
            if index in used:
                continue
            try:
                matches = abs(float(value) - float(target)) <= tolerance
            except (TypeError, ValueError):
                matches = False
            if matches:
                used.add(index)
                hits += 1
                break
    precision, recall = hits / len(actual), hits / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score_q5_field(actual: Any, expected: Any, spec: dict[str, Any]) -> float:
    if expected == "Unreliable" or expected is None:
        return float(actual is None or str(actual).lower() in {"unreliable", "null", "none"})
    if actual is None:
        return 0.0
    kind = spec.get("type")
    if kind == "bool":
        if isinstance(actual, str) and actual.lower() in {"true", "false"}:
            actual = actual.lower() == "true"
        elif isinstance(actual, int) and actual in (0, 1):
            actual = bool(actual)
        return float(isinstance(actual, bool) and actual == expected)
    if kind == "enum":
        return _enum_score(actual, expected)
    if kind in {"int_count", "count_float"} and (isinstance(actual, list) or isinstance(expected, list)):
        actuals = actual if isinstance(actual, list) else [actual]
        expecteds = expected if isinstance(expected, list) else [expected]
        if not actuals or not expecteds:
            return float(not actuals and not expecteds)
        matched = [score_q5_field(a, e, {**spec, "type": "abs"}) for a, e in zip(actuals, expecteds)]
        return sum(matched) / max(len(actuals), len(expecteds))
    if kind in {"int_count", "count_float", "abs"}:
        try:
            delta = abs(float(actual) - float(expected))
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if delta <= spec["tol_full"] else 0.5 if delta <= spec["tol_half"] else 0.0
    if kind == "pct":
        try:
            actual_value, expected_value = float(actual), float(expected)
        except (TypeError, ValueError):
            return 0.0
        ratio = abs(actual_value - expected_value) / abs(expected_value) if expected_value else abs(actual_value)
        return 1.0 if ratio <= spec["tol_full"] else 0.5 if ratio <= spec["tol_half"] else 0.0
    if kind == "coord_list":
        if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
            return 0.0
        values = [score_q5_field(a, e, {**spec, "type": "abs"}) for a, e in zip(actual, expected)]
        return sum(values) / len(values) if values else 1.0
    if kind == "array_int_match":
        return _array_f1(actual, expected, spec["tol_full"])
    if kind == "array_float_match":
        if not isinstance(actual, list) or not isinstance(expected, list):
            return 0.0
        if not expected:
            return 1.0
        if not actual:
            return 0.0
        scores = []
        for a, e in zip(actual, expected):
            try:
                actual_value, expected_value = float(a), float(e)
                if expected_value == 0:
                    scores.append(float(abs(actual_value) < 0.01))
                else:
                    ratio = abs(actual_value - expected_value) / abs(expected_value)
                    scores.append(float(ratio <= spec["tol_full"]))
            except (TypeError, ValueError):
                scores.append(0.0)
        return sum(scores) / max(len(actual), len(expected))
    return 0.0


def score_reproducible_subset(entry: dict[str, Any], responses: dict[str, Any]) -> dict[str, Any]:
    ground_truth = entry.get("ground_truth", {})
    answer = lambda q: (responses.get(QUESTION_NAMES[q], {}) or {}).get("answer")

    q2_actual, q2_expected = extract_classification(answer("Q2")), extract_classification(ground_truth.get("Q2"))
    q4_actual, q4_expected = extract_assessment(answer("Q4")), extract_assessment(ground_truth.get("Q4"))
    q6_actual, q6_expected = extract_status(answer("Q6")), extract_status(ground_truth.get("Q6"))
    aliases = {"TOO_MANY_OSC": {"SAMPLING_TOO_COARSE"}, "SAMPLING_TOO_COARSE": {"TOO_MANY_OSC"}}
    q6_correct = bool(q6_actual and q6_expected and (q6_actual == q6_expected or q6_actual in aliases.get(q6_expected, set())))

    expected_q5 = parse_json_answer(ground_truth.get("Q5"))
    actual_q5 = parse_json_answer(answer("Q5"))
    q5_fields = {}
    if isinstance(expected_q5, dict):
        for field, spec in entry.get("q5_scoring", {}).items():
            if field in expected_q5:
                q5_fields[field] = score_q5_field(
                    actual_q5.get(field) if isinstance(actual_q5, dict) else None,
                    expected_q5[field], spec,
                )
    q5_score = 100 * sum(q5_fields.values()) / len(q5_fields) if q5_fields else None
    return {
        "Q1": {"score": None, "reason": "official score includes an LLM key-point judge"},
        "Q2": {"score": 100.0 if q2_actual and q2_actual == q2_expected else 0.0, "actual": q2_actual, "expected": q2_expected},
        "Q3": {"score": None, "reason": "official score requires an LLM key-point judge"},
        "Q4": {"score": 100.0 if q4_actual and q4_actual == q4_expected else 0.0, "actual": q4_actual, "expected": q4_expected},
        "Q5": {"score": round(q5_score, 1) if q5_score is not None else None, "fields": q5_fields},
        "Q6": {"score": 100.0 if q6_correct else 0.0, "actual": q6_actual, "expected": q6_expected},
    }
