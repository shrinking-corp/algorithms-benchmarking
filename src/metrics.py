"""
CodeBLEU wrapper with fallback to token-overlap.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from codebleu import calc_codebleu

    _CODEBLEU_AVAILABLE = True
except ImportError:
    _CODEBLEU_AVAILABLE = False
    logger.warning(
        "Package 'codebleu' is not installed. "
        "Using token-overlap as a fallback metric. "
        "Install with: pip install codebleu"
    )


def calculate_codebleu(reference: str, hypothesis: str, language: str = "python") -> float:
    """
    Calculates CodeBLEU score between reference and generated code.

    Returns:
        float in [0, 1] - 1.0 = perfect match
    """
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    if _CODEBLEU_AVAILABLE:
        try:
            result = calc_codebleu(
                references=[[reference]],
                predictions=[hypothesis],
                lang=language,
                weights=(0.25, 0.25, 0.25, 0.25),
            )
            return float(result["codebleu"])
        except Exception as exc:
            logger.warning("CodeBLEU calculation failed (%s), using fallback.", exc)

    return _token_overlap(reference, hypothesis)


def _token_overlap(reference: str, hypothesis: str) -> float:
    """Simple token-overlap (precision) used as a fallback."""
    ref_tokens = set(reference.split())
    hyp_tokens = set(hypothesis.split())
    if not hyp_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(hyp_tokens)
