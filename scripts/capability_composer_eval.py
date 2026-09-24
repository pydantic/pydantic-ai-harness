"""Hand-run check of `CapabilityComposer` picks against labelled prompts.

Not collected by pytest and not wired into CI: it calls the picker live, which costs a request per prompt
(and, for the default Jev picker, needs a TypeSafe key). Run it after changing the model menu, a catalog
description, or the picker or its version:

    TYPESAFE_API_KEY=... uv run python scripts/capability_composer_eval.py
    TYPESAFE_API_KEY=... uv run python scripts/capability_composer_eval.py --picker-model typesafe:jev-1.13.0 --repeat 3
    uv run python scripts/capability_composer_eval.py --picker-model openai-codex:gpt-6-luna

It asks the picker about every prompt in `capability_composer_eval_prompts.txt` with the docs' example menu and the default
catalog, and reports per section:

- how often the model pick matches the label, and how often prompts labelled `none` fall through,
- how often a handed-off prompt got every capability its label needs, per tier, and which were missed,
- for each `confidence_threshold`, how the unsure picks would land with each fallback: every `models` key as
  `unsure_model`, or one tier above the picker's pick. `under` counts runs on a weaker model than the label.

Only the picker is called; no sub-agent runs. Tune against part of the set and check against the rest, or the
numbers describe the prompts rather than the composer.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai_harness.capability_composer import CapabilityComposer, Composition, default_catalog
from pydantic_ai_harness.subagents import ModelOption

PROMPTS = Path(__file__).with_name('capability_composer_eval_prompts.txt')
MENU = {
    'fast': ModelOption(
        'openai-codex:gpt-6-luna',
        description='Answering a question, or one command or trivial edit that needs no investigation',
    ),
    'medium': ModelOption(
        'openai-codex:gpt-6-sol',
        description='A focused code change with a clear cause or spec, in one or a few files',
    ),
    'max': ModelOption(
        'openai-codex:gpt-6-astra',
        description='Open-ended work: an unknown root cause, a design decision, or changes across many files',
    ),
}
TIERS = list(MENU)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
ONE_UP = 'one-up'


@dataclass(frozen=True)
class Case:
    """One labelled prompt."""

    section: str
    tier: str
    needs: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class Outcome:
    """What the picker picked for a case."""

    case: Case
    composition: Composition

    @property
    def fell_through(self) -> bool:
        """The picker picked no capabilities, so the main agent keeps the turn."""
        return not self.composition.capabilities

    @property
    def equipped(self) -> bool:
        """Every capability the label needs was picked."""
        return all(key in self.composition.capabilities for key in self.case.needs)


def load(path: Path) -> list[Case]:
    """Parse the prompts file."""
    cases: list[Case] = []
    section = 'all'
    for line in path.read_text().splitlines():
        if line.startswith('# section:'):
            section = line.removeprefix('# section:').strip()
        if not line.strip() or line.startswith('#'):
            continue
        tier, needs, prompt = line.split('|', 2)
        if tier != 'none' and tier not in MENU:
            raise ValueError(f'Unknown tier {tier!r} in: {line}')
        cases.append(Case(section, tier, tuple(key for key in needs.split(',') if key), prompt))
    return cases


async def ask(composer: CapabilityComposer[object], cases: list[Case], repeat: int, concurrency: int) -> list[Outcome]:
    """Ask the picker about every case `repeat` times, with at most `concurrency` requests in flight."""
    queue = iter([case for case in cases for _ in range(repeat)])
    outcomes: list[Outcome] = []

    async def worker() -> None:
        for case in queue:
            outcomes.append(Outcome(case, await composer.compose(case.prompt)))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return outcomes


def landed(outcome: Outcome, threshold: float, fallback: str) -> str:
    """The `models` key the sub-agent runs on, given where unsure picks go."""
    picked = outcome.composition.model
    if outcome.composition.confidence.get('model', 1.0) >= threshold:
        return picked
    if fallback == ONE_UP:
        return TIERS[min(TIERS.index(picked) + 1, len(TIERS) - 1)]
    return fallback


def share(part: int, whole: int) -> str:
    """`part/whole (percent)`."""
    return f'{part}/{whole} ({part / whole:.0%})' if whole else '-'


def report(name: str, outcomes: list[Outcome]) -> None:
    """Print one section's numbers."""
    coding = [o for o in outcomes if o.case.tier != 'none']
    chat = [o for o in outcomes if o.case.tier == 'none']
    handed = [o for o in coding if not o.fell_through]
    print(f'\n## {name}: {len(outcomes)} calls')
    print(
        f'model pick matches label: {share(len([o for o in coding if o.composition.model == o.case.tier]), len(coding))}'
    )
    if chat:
        print(f'`none` prompts fell through: {share(len([o for o in chat if o.fell_through]), len(chat))}')
    print(f'coding prompts fell through: {share(len(coding) - len(handed), len(coding))}')
    for tier in TIERS:
        in_tier = [o for o in handed if o.case.tier == tier]
        print(f'got every needed capability, {tier}: {share(len([o for o in in_tier if o.equipped]), len(in_tier))}')
    missed = Counter(key for o in handed for key in o.case.needs if key not in o.composition.capabilities)
    print(f'missed capabilities: {dict(missed.most_common())}')
    print('unsure picks by fallback (exact / under / over / escalated):')
    for threshold in THRESHOLDS:
        cells: list[str] = []
        for fallback in [*TIERS, ONE_UP]:
            ran = [(TIERS.index(landed(o, threshold, fallback)), TIERS.index(o.case.tier)) for o in handed]
            exact = len([1 for got, want in ran if got == want])
            under = len([1 for got, want in ran if got < want])
            cells.append(f'{fallback} {exact}/{under}/{len(ran) - exact - under}')
        unsure = len([o for o in handed if o.composition.confidence.get('model', 1.0) < threshold])
        print(f'  t={threshold}: ' + ' | '.join(cells) + f'  escalated {unsure}')


async def main() -> None:
    """Run the check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--picker-model', default='typesafe:jev-latest')
    parser.add_argument('--repeat', type=int, default=1, help='ask about each prompt this many times')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--section', help='only run this section')
    args = parser.parse_args()
    if args.concurrency < 1 or args.repeat < 1:
        parser.error('--concurrency and --repeat must be at least 1')

    cases = [case for case in load(PROMPTS) if args.section in (None, case.section)]
    composer = CapabilityComposer[object](models=MENU, catalog=default_catalog(), picker_model=args.picker_model)
    outcomes = await ask(composer, cases, args.repeat, args.concurrency)
    for section in dict.fromkeys(case.section for case in cases):
        report(section, [o for o in outcomes if o.case.section == section])
    print('\nlowest-confidence picks:')
    for o in sorted(outcomes, key=lambda o: o.composition.confidence.get('model', 1.0))[:10]:
        confidence = o.composition.confidence.get('model', 1.0)
        print(f'  {confidence:.2f} label={o.case.tier} picked={o.composition.model} :: {o.case.prompt}')


if __name__ == '__main__':
    asyncio.run(main())
