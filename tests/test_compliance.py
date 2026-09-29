import pytest

from sales_reply_assistant.compliance import find_compliance_issues


@pytest.mark.parametrize(
    "text",
    [
        "Этот набор вылечит ваш гастрит.",
        "Хлорелла лечит анемию.",
        "Гарантируем результат уже через неделю!",
        "Можно принимать вместо лекарств.",
        "Смело отмените приём препаратов.",
        "Это лекарство нового поколения.",
    ],
)
def test_flags_medical_claims(text: str) -> None:
    assert find_compliance_issues(text)


@pytest.mark.parametrize(
    "text",
    [
        "Наши продукты не лечат заболевания и не заменяют терапию.",
        "Набор не может вылечить гастрит, обсудите приём с врачом.",
        "Мы не можем обещать, что продукты лечат болезни.",
        "Я не могу заявлять, что набор лечит все болезни.",
        "Доставка займёт 3–7 дней, оплатить можно при получении.",
        "Продукты не предназначены для лечения заболеваний.",
    ],
)
def test_accepts_safe_wording(text: str) -> None:
    assert find_compliance_issues(text) == []


def test_negation_does_not_leak_into_the_next_sentence() -> None:
    text = "Мы не ставим диагнозы. Этот набор вылечит гастрит."
    assert find_compliance_issues(text)


def test_reports_each_rule_once() -> None:
    issues = find_compliance_issues("Вылечит за неделю, вылечит навсегда.")
    assert len(issues) == 1
