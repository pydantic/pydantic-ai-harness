---
name: final-review
description: Run a final review pass on a harness PR that has already been iterated on - bots have posted, threads have been resolved, and commits have landed since the first review. Use it before asking a maintainer to merge, or when a PR has changed a lot since it was last read. Do NOT use it as a first read of a fresh PR: its highest-value charters audit what earlier rounds claimed, so with no earlier rounds most of it has nothing to work on. Do NOT use it to re-report per-line correctness or security findings - Macroscope and Veria already post those on every PR.
---

# Final review

Audit an iterated harness PR: what earlier rounds claimed, what the diff grew
into, and what nobody ran.

Three rules decide everything below.

**Audit the claims first.** The claim charters find more per token than anything
else here, because a claim someone already made and nobody rechecked is invisible
to every bot on the PR.

**Reproduce a claim before accepting it and before dismissing it.** This applies
to bot findings, to resolved threads, to review replies, to tracking issues, and
to anything a previous session wrote down.

**A charter declares what would falsify it.** A charter that runs nothing and
reads nothing cannot fail, so it always reports clean.

Six charters cost roughly 600k subagent tokens. Choose three when the PR is
small. Choose six when it has a long iteration history.

## Phase 0 - ground

1. `git fetch origin main`. Record whether the branch is behind, and by how much.
2. If the branch is behind, establish every claim about main with
   `git show origin/main:<file>`. Never read main from the merge-base. A stale
   worktree produces findings about code main already changed.
3. Build the diff once. Strip cassettes.
4. List the iteration history: resolved review threads, dismissed bot findings,
   commits pushed since the first review, and the verification claims wherever
   they live - a PR body, a review reply, a tracking issue, or a local notes file.
   This list is the input to the claim charters.

Then run the mechanical pre-pass:

- Every pinned `uses:` SHA in a changed workflow resolves:
  `gh api repos/<owner>/<repo>/commits/<sha>`.
- A guarantee sentence changed in one doc surface is changed in the other.
  Compare the added clause, not the changed line: `docs/` uses site-relative
  links where a README uses plain backticks, so a line diff reports differences
  that are repo convention.
- No test file imports a `_`-prefixed helper from another module. Check every
  test file the diff touches, not only new ones.
- The diff adds no `Any`, no `cast`, and no `getattr` by string name on a type
  the repo declares.
- Every reviewer that normally posts on a harness PR posted one. Establish the
  norm by sampling recent merged PRs rather than assuming. A quota notice is not
  a review, and a skipped verdict is not an approval.

## Phase 1 - charter

Write 2 to 6 charters. Each states the concern, its lane, the files it covers,
and what would falsify it. Reject a charter that cannot state what would falsify
it.

### Claim charters - write these first

Always write at least one when the iteration history is not empty. Audit:

- a finding a bot marked resolved. Open the resolving commit and confirm it
  addresses the finding
- a finding someone agreed to and never applied
- a thread resolved without a verifiable fix
- **every verification claim this PR's own author made**, in the body, in review
  replies, and in any issue the PR opened. An author who ran the right command
  and then wrote the wrong sentence is the single most common finding here, and
  no bot on the PR can see it

### Lane A - the charter runs code

Use Lane A when two versions of the code can disagree on a printed result. A
Lane A charter names the two versions and the input set.

- **Guard grammar against consumer grammar.** A validator rejects what the
  downstream accepts, or forwards what it rejects. Check both directions.
- **Version and OS cells.** A branch that exists on one Python version or one OS
  is untested on the others. Coverage stays at 100% while a cell is wrong.
- **Vacuity.** See the escalation rule below.
- **Cap accounting.** Measure the limit on the final value the caller receives,
  after framing, truncation markers, envelopes and metadata.
- **Option to consumer.** Run once per value of a new option. Print which branch
  of the consumer each value reached.
- **Positional signature compatibility.** Construct the changed dataclass
  positionally with the previous argument list. Print what binds to what.
- **Resolution compatibility.** `pydantic-ai-harness` and `pydantic-ai-slim` ship
  separately. Resolve under `--locked`, `--resolution lowest-direct`, and latest,
  and print the resolved versions.
- **Durability composition.** A capability that holds state or overrides
  `for_run` needs a public `Agent` test per supported wrapper. Assert state
  survives a replay boundary, or assert the composition fails before the first
  tool call.
- **A measured number in the docs.** A doc sentence quoting a benchmark, a count,
  or a limit is settled by running it, not by reading it.
- **Prove the residual you are deferring is not yours.** When the PR defers a
  known gap, A/B it against main. A residual that main already had is a defer.
  A residual this diff introduced is a finding.

### Lane B - the charter reads

Use Lane B when no probe can settle the concern. Do not invent a probe to make a
Lane B charter look like Lane A. **Lane B produces most of the findings in this
skill. Give it the same care as Lane A.**

- **Prose against code.** For every guarantee sentence the diff adds or leaves
  standing in a README, a `docs/` page, or a docstring: is the sentence true of
  the code as written? Read the implementation the sentence describes and name
  the line that limits it. Docs parity tests check that both surfaces exist and
  match each other. No test checks whether either is true. In this repo the
  documented guarantee is consistently stronger than the code, never weaker.
- **Workflow graph semantics.** `needs` against a skipped dependency, `always()`
  against `!cancelled()`, the permission ceiling a caller imposes on a reusable
  workflow, a job-level `environment` shadowing a `workflow_call` secret. Decide
  these from the workflow file. Reproducing them means waiting on a schedule.
- **Cross-PR interference.** Name every open PR that edits the same function.
  State which mechanism becomes redundant, whose comment rationale goes stale,
  and what a test-merge conflicts on.
- **Boundary and naming.** Whether the behavior belongs in core rather than
  harness, whether the capability name follows the repo's naming rule, and
  whether a link to an open issue carries the required comment on that issue.

## Phase 2 - investigate

Dispatch one agent per charter. Instruct each to falsify its charter. A charter
falsified with evidence is a complete review that found nothing. An invented
finding is the failure this instruction prevents.

Give each agent no conclusions and every environment constraint. It does not
know `UV_FROZEN=1`, the scratchpad path, that `ALLOW_MODEL_REQUESTS = False`
blocks provider calls, or that it must never run the full suite. Withholding
your conclusions is the point; withholding the environment just costs a retry.

### Isolation - read before choosing a primitive

**A probe that mutates the working tree cannot run in parallel with any other
charter.** Concurrent charters that mutate a shared tree corrupt each other's
results, and a corrupted result looks exactly like a real finding.

So: every mutating probe gets its own isolated tree, or runs serially with no
other charter in flight. Isolate with `git worktree add`, or export a revision
with `git archive <rev> | tar -x -C <dir>` and point `PYTHONPATH` at it.

Primitives, strongest first:

1. **Isolated tree.** `git worktree add` or `git archive` export. Safe under
   parallel dispatch. Use this by default.
2. **Two live modules in one process.** Load main's module beside the branch's
   with `importlib.util.spec_from_file_location`, then run one input through
   both. Safe: it reads files, it does not write them.
3. **Case matrix.** N inputs by M versions. One row per input, flag the rows that
   differ. One field or key per row - an aggregate comparison hides which side
   owns the difference.
4. **Control arm.** Add an arm that must not fail, so a failure pins to one
   component.
5. **Source swap** (`git show origin/main:<file> > <file>`, run, restore) and
   **scratch patch**. Both mutate. Serial only.

**Every probe asserts its own provenance before reporting a number.** Print the
resolved module path and the revision under test. `python -m pytest` puts the
current directory on `sys.path` ahead of `PYTHONPATH`, so a probe can silently
measure a tree it did not intend to.

### The vacuity charter

Revert-and-rerun alone is not enough, and taking it literally deletes good tests.

1. Revert the fix and rerun. A test that fails is doing its job.
2. A test that still passes is **not yet** vacuous. It may be the positive
   control half of a boundary pair.
3. Distinguish ERROR from FAIL. Reverting a file that lacks a new symbol makes
   tests ERROR on ImportError, which only proves the symbol is new.
4. Escalate to a targeted mutation: change the boundary by one, neutralize the
   helper, and rerun. Only a test that survives mutation guards nothing.

### When a charter dies

Try resuming it before reporting it failed. Infrastructure failures are usually
resumable. Report a charter that cannot be resumed as failed - an incomplete
review is the honest verdict, not a reason to keep retrying.

## Phase 3 - adjudicate

**Adjudicate in-session by default.** You have just read every charter result in
full, so judging provenance and evidence quality is your job, not a round trip.
Dispatch a separate adjudicator only when you authored the code under review and
want an independent verdict.

Assign each finding one verdict:

- **CONFIRMED** - the evidence holds and the finding is about this diff
- **RESCOPED** - the behavior is real but main owns it, not this diff
- **REFUTED** - positive contradicting evidence exists
- **UNSETTLED** - the evidence is absent

REFUTED requires evidence that contradicts the finding. Absence of confirming
evidence is UNSETTLED. A wrongly refuted finding is deleted silently.

A RESCOPED finding needs a tracking artifact. Open it in this run when you have
write access. When the caller invoked this pass as report-only, name the artifact
that should be opened and let them authorize it.

Settle provenance by running both versions. This repo has no blame map in the
review path.

## Phase 4 - report

Give every finding a severity and a verdict. They are independent:

- severity is BLOCKING or WARNING
- verdict is `do`, `defer`, or `skip`, judged on net value, with its reason on
  the same line

A WARNING whose verdict is `do` is a required action.

Where the report goes is the caller's call. Posting one comment per run is the
default when the caller wants it on the PR. Do not edit an earlier run's comment:
a final review audits what earlier rounds said, so earlier rounds stay readable.

Report a charter that failed as failed.

## Constraints on your own behavior

- Never run repository-wide pyright, pytest, or coverage. Run targeted files and
  nodes.
- Treat 100% branch coverage as a property of combined matrix data. A local gap
  is not evidence.
- Write no em dashes, no superlatives, and no marketing adjectives in anything
  you post.
- Change no file on the branch under review.
