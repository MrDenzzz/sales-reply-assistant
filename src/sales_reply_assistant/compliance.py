"""Deterministic safety net for the client reply.

Dietary supplements must not be advertised as medicine (Russian Federal Law "On Advertising",
art. 25). The system prompt already forbids such claims; this stop-list catches the cases where
the model slips anyway, so the manager sees a warning before sending the reply.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComplianceRule:
    pattern: re.Pattern[str]
    warning: str


RULES: tuple[ComplianceRule, ...] = (
    ComplianceRule(
        re.compile(r"\b(?:вы|из)?леч(?:ит|ат|ить|ите|ится)\b|\bисцел\w*", re.I),
        "Обещание лечебного эффекта: БАД не лечит, переформулируйте.",
    ),
    ComplianceRule(
        re.compile(r"\bгарантир\w*\s+(?:результат|эффект|излечени|выздоровлени)", re.I),
        "Гарантия результата: для БАД это недопустимое обещание.",
    ),
    ComplianceRule(
        re.compile(
            r"\b(?:вместо|замен\w*)\s+(?:лекарств|препарат|лечени)\w*"
            r"|\b(?:отмен\w*|брос\w*|прекрат\w*)\s+(?:при[её]м\s+)?(?:лекарств|препарат)\w*",
            re.I,
        ),
        "Совет заменить лечение или отменить препараты — так писать нельзя.",
    ),
    ComplianceRule(
        re.compile(r"\bэто\s+(?:лекарство|препарат|медикамент)\b", re.I),
        "Продукт назван лекарством — БАД не является лекарственным средством.",
    ),
)

# "не лечит", "не может вылечить" — negation right before the claim or one word earlier.
_DIRECT_NEGATION = re.compile(r"\b(?:не|ни|нельзя)\s+(?:\w+\s+)?$", re.I)
# "мы не можем обещать, что продукты лечат" — the whole clause is under a negated verb.
_NEGATED_CLAUSE = re.compile(
    r"\bне\s+(?:мо[жг]\w*|способн\w*|обеща\w*|гарантир\w*|утвержда\w*|заявля\w*|предназначен\w*)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"[.!?\n]")


def find_compliance_issues(text: str) -> list[str]:
    """Return a warning for every rule the text violates (negated phrases are fine)."""
    return [
        rule.warning
        for rule in RULES
        if any(not _is_negated(text, match.start()) for match in rule.pattern.finditer(text))
    ]


def _is_negated(text: str, position: int) -> bool:
    prefix = text[:position]
    sentence_start = max((m.end() for m in _SENTENCE_END.finditer(prefix)), default=0)
    sentence = prefix[sentence_start:]
    return bool(_DIRECT_NEGATION.search(sentence) or _NEGATED_CLAUSE.search(sentence))
