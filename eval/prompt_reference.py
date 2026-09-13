#!/usr/bin/env python3
"""Zero-training metric-reference prompt baselines.

The presets expose only public field dimensions.  They deliberately contain no
sample-specific coordinates, counts, or answers, so the intervention tests
whether an explicit metric ruler explains gains attributed to target-domain RL.
"""
import hashlib


REFERENCE_PRESETS = {
    "none": "",
    "basketball_28x15": (
        "Metric reference only: a regulation full basketball playing court is "
        "28.0 metres long and 15.0 metres wide. Use those dimensions only as a "
        "visual scale reference; infer the requested objects and answer from the image."
    ),
    "soccer_105x68": (
        "Metric reference only: the reference association-football pitch is "
        "105.0 metres long and 68.0 metres wide. Use those dimensions only as a "
        "visual scale reference; infer the requested objects and answer from the image."
    ),
}


def load_reference_text(preset="none", custom_text=None):
    """Return ``(tag, text)`` for auditable inference/resume metadata."""
    if custom_text is not None:
        text = custom_text.strip()
        if not text:
            raise ValueError("custom reference text is empty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"custom_sha256_{digest}", text
    if preset not in REFERENCE_PRESETS:
        raise ValueError(f"unknown reference preset: {preset}")
    return preset, REFERENCE_PRESETS[preset]


def compose_reference_prompt(question, reference_text):
    """Prepend the same non-answer reference to every benchmark question."""
    question = str(question).strip()
    reference_text = (reference_text or "").strip()
    if not reference_text:
        return question
    return f"{reference_text}\n\nQuestion: {question}"

