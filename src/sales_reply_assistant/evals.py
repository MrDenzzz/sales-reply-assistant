"""Evaluation harness: runs scripted dialogs against the real model and checks the behaviour.

Usage::

    uv run sales-reply-eval                      # all cases once
    uv run sales-reply-eval --runs 3 --show      # stability check, print drafts
    uv run sales-reply-eval --only pregnancy
"""

import argparse
import asyncio
import logging
import re
import statistics
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from sales_reply_assistant.config import Settings
from sales_reply_assistant.errors import AssistantError
from sales_reply_assistant.schemas import DialogMessage, Suggestion, SuggestRequest
from sales_reply_assistant.services import build_services


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_attention: bool | None = None
    upsell_should_offer: bool | None = None
    sources_include: list[str] = Field(default_factory=list)
    sources_any: list[str] = Field(default_factory=list)
    products_any: list[str] = Field(default_factory=list)
    reply_contains_any: list[str] = Field(default_factory=list)
    reply_excludes: list[str] = Field(default_factory=list)
    upsell_excludes: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    client_name: str | None = None
    messages: list[DialogMessage]
    expect: Expectations

    def to_request(self) -> SuggestRequest:
        return SuggestRequest(messages=self.messages, client_name=self.client_name)


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[EvalCase]).validate_python(raw)


def check(suggestion: Suggestion, case: EvalCase) -> list[str]:
    """Return human-readable failures; an empty list means the case passed."""
    expect = case.expect
    failures: list[str] = []
    if case.client_name is None and (name := _invented_name(suggestion.reply)):
        failures.append(f"reply addresses the client as {name!r}, but the name is unknown")
    if expect.needs_attention is not None and suggestion.needs_attention != expect.needs_attention:
        failures.append(
            f"needs_attention={suggestion.needs_attention}, expected {expect.needs_attention}"
        )
    if (
        expect.upsell_should_offer is not None
        and suggestion.upsell.should_offer != expect.upsell_should_offer
    ):
        failures.append(
            f"upsell.should_offer={suggestion.upsell.should_offer}, "
            f"expected {expect.upsell_should_offer}"
        )

    source_ids = {source.id for source in suggestion.sources}
    missing = [source for source in expect.sources_include if source not in source_ids]
    if missing:
        failures.append(f"sources lack {missing}, got {sorted(source_ids)}")
    if expect.sources_any and not set(expect.sources_any) & source_ids:
        failures.append(f"sources {sorted(source_ids)} contain none of {expect.sources_any}")

    product_ids = [product.id for product in suggestion.upsell.products]
    if expect.products_any and not set(expect.products_any) & set(product_ids):
        failures.append(f"products {product_ids} contain none of {expect.products_any}")

    reply = suggestion.reply.lower()
    if expect.reply_contains_any and not any(s.lower() in reply for s in expect.reply_contains_any):
        failures.append(f"reply mentions none of {expect.reply_contains_any}")
    failures.extend(
        f"reply must not contain {s!r}" for s in expect.reply_excludes if s.lower() in reply
    )
    upsell = f"{suggestion.upsell.hint} {suggestion.upsell.pitch}".lower()
    failures.extend(
        f"upsell must not contain {s!r}" for s in expect.upsell_excludes if s.lower() in upsell
    )
    failures.extend(f"compliance: {warning}" for warning in suggestion.compliance_warnings)
    return failures


_ADDRESS = re.compile(r"^([А-ЯЁ][а-яё]+),")
_GREETINGS = {"Здравствуйте", "Привет", "Добрый", "Спасибо", "Извините", "Простите", "Понимаю"}


def _invented_name(reply: str) -> str | None:
    """A reply that opens with "Марина, ..." when the client's name was never given."""
    match = _ADDRESS.match(reply)
    if match and match.group(1) not in _GREETINGS:
        return match.group(1)
    return None


async def run(cases: list[EvalCase], settings: Settings, runs: int, show: bool) -> bool:
    latencies: list[int] = []
    passed = total = 0
    async with build_services(settings) as services:
        print(f"model: {services.chat_model}, embeddings: {services.embedding_model}\n")
        for run_number in range(1, runs + 1):
            if runs > 1:
                print(f"--- run {run_number}/{runs}")
            for case in cases:
                total += 1
                try:
                    suggestion = await services.assistant.suggest(case.to_request())
                except AssistantError as error:
                    print(f"FAIL  {case.id:<22} error: {error}")
                    continue
                latencies.append(suggestion.latency_ms)
                failures = check(suggestion, case)
                passed += not failures
                verdict = "PASS" if not failures else "FAIL"
                print(
                    f"{verdict}  {case.id:<22} {suggestion.latency_ms / 1000:5.1f} s  {case.title}"
                )
                for failure in failures:
                    print(f"        - {failure}")
                if show:
                    _print_suggestion(suggestion)

    median = statistics.median(latencies) / 1000 if latencies else 0
    print(f"\npassed {passed}/{total}, median latency {median:.1f} s")
    return passed == total


def _print_suggestion(suggestion: Suggestion) -> None:
    print(f"        intent:    {suggestion.client_intent}")
    print(f"        attention: {suggestion.needs_attention} {suggestion.attention_reason or ''}")
    print(f"        sources:   {[source.id for source in suggestion.sources]}")
    print(f"        reply:     {suggestion.reply}")
    products = [product.id for product in suggestion.upsell.products]
    print(f"        upsell:    offer={suggestion.upsell.should_offer} {products}")
    print(f"        hint:      {suggestion.upsell.hint}")
    print(f"        pitch:     {suggestion.upsell.pitch}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behaviour evals against the real model.")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.yaml"))
    parser.add_argument("--runs", type=int, default=1, help="repeat to check stability")
    parser.add_argument("--only", nargs="+", metavar="CASE_ID", help="run selected cases only")
    parser.add_argument("--show", action="store_true", help="print drafts and hints")
    args = parser.parse_args()

    # With --show, the assistant's log reveals the retrieved context of every case.
    logging.basicConfig(level=logging.INFO if args.show else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cases = load_cases(args.cases)
    if args.only:
        cases = [case for case in cases if case.id in args.only]
    ok = asyncio.run(run(cases, Settings(), args.runs, args.show))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
