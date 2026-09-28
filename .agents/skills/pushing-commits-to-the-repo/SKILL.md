---
name: pushing-commits-to-the-repo
description: Open and advance a PR -- write current metadata, run an independent pre-push review,
  push, watch CI, and triage every comment. Use whenever you open a PR or push a commit to one.
---

# pushing-commits-to-the-repo

Pushing starts a loop; it does not end the task. **Work stops only when CI is green, at least one
hosted AI review has finished on the current HEAD, AND no comment is left unresolved.**

Lifecycle: implement -> targeted verification -> commit -> independent pre-push review -> remediate
and re-review when required -> push -> full CI and coverage -> hosted reviewers -> docs parity when
applicable -> final metadata check.

## When you open the PR

### Write the title and body

Follow the title and template rules in the root `AGENTS.md`.

Keep visible body content within 40 lines. Exclude template lines and collapsed `<details>`
contents from the count. For a feature or behavior change, use this order:

1. **Why we make these changes** -- State the problem and decision in a few sentences. Link the issue.
2. **New public surface** -- List each new maintained symbol. Write `none` when there is none.
3. **User-visible behavior** -- Show the smallest before-and-after example. Replace it with a
   call-path diff when the changed call chain explains the behavior; do not include both.
4. **Verification** -- Link the exact proving tests from the PR's Files changed tab so links survive
   later pushes. Put a minimal runnable playground in `<details>` only when it helps reviewers
   reproduce the behavior.
5. **What changes for existing users** -- State the effect in one sentence. `Nothing` is valid.

Use one collapsed `<details>` section per goal only when the PR has multiple independent goals.
For a trivial PR, use the issue link, a short summary, and the test plan.

#### User-visible call-path diff

Use one fenced `diff` tree from the public entry point to the changed observable result.

- Format each node as `path/file.py :: Class.method()` or `path/file.py :: function()`.
- Indent each callee beneath its caller with `+-`. Preserve enough unchanged nodes to show each edge.
- Collapse irrelevant intermediate calls as `... unchanged machinery ...`.
- Include arguments only when they explain the change.
- Include results only on relevant leaves.
- Keep the shared caller prefix unmarked. Mark only diverging nodes, relevant arguments, or results.
- Target 12 content lines inside the fence. Never exceed 20; collapse secondary branches instead.

Apply a label -- the repo triages and filters by them. Fetch the real list first with
`gh label list --limit 100`, because the set changes and a guessed label silently fails to apply.
Pick the one naming what the PR *is* (`bug`, `enhancement`, `documentation`) and add a topic label
where one fits (`capability`, `primitive`, `agent-feature`, `core-change`, `code-mode`, `media`,
`externalization`, ...): `gh pr edit <number> --add-label <label>`.

Labelling needs triage permission on the repo. If it fails, quote the actual error rather than
concluding you lack permission.

## Before you push -- independent review gate

Run this gate before every push. It guarantees context independence, not hosted-grade hostile-content
isolation, and catches semantic defects before they consume a CI and hosted-review round.

1. Run targeted verification while iterating, then run the root `AGENTS.md` mandatory pre-commit
   checks before committing the exact state intended for push. Leave nothing staged, unstaged, or
   uncommitted unless the user's instructions override this.
2. Fetch the declared target branch. Capture and validate three full SHAs: the current target tip as
   `policy-base-sha`, its merge base with the candidate as `merge-base-sha`, and the exact candidate
   commit as `candidate-head-sha`. Verify the candidate worktree is clean.
3. From a checkout pinned to the policy-base SHA, prepare the review bundle: task or issue, full PR
   discussion including thread state, relevant settled maintainer decisions with their sources,
   relevant authoritative documentation, completed verification, and the exact
   merge-base-to-candidate diff with external diff and text conversion disabled.
   If this candidate introduces `pre-push-review` and the stable checkout has no copy, use the
   stable root review rubric and instructions while treating the candidate skill as review material.
   This bootstrap exception ends once the skill lands on the target branch.
4. Launch the strongest locally available reviewer from that stable checkout in a fresh subagent
   with no inherited conversation. Have it follow the stable checkout's `pre-push-review` skill, or
   the stable root review rubric and instructions during the bootstrap exception.
   Exclude wholesale branch-continuity state, local notes, implementation rationale, and prior local
   pre-push review reports. Treat the supplied settled decisions as constraints and assess
   conformance instead of reopening them. Restrict it to its harness's native read and search tools;
   tool availability is not the independence guarantee.
   If no eligible reviewer can be launched, the gate fails; do not push.
5. Require actionable findings or `current at <full-candidate-head-sha>`. Triage every finding.
   Remediate valid findings and run the required verification before committing; dismiss invalid
   findings only with concrete evidence. If a finding exposes a real design choice, API trade-off,
   or behavioral default, pause the push and give the maintainer the options, trade-offs, evidence,
   and a recommendation; record the resulting decision. Dispatch a different fresh subagent only
   when remediation changes the candidate HEAD or a maintainer decision changes the acceptance
   criteria. An evidence-backed dismissal with an unchanged candidate HEAD does not require another
   pass. Escalate persistent disagreement.

Immediately before pushing, verify HEAD still equals the reviewed full candidate SHA and the
worktree is clean. Any mismatch restarts the gate.

Never use the implementing agent as the reviewer. Never treat this gate as test execution.

Never force-push an open PR branch. Push follow-up commits so previous reviews remain valid;
maintainers can squash them when merging.

Attempt the push. If it fails, read the real error. Do not infer a restriction from metadata.

## After you push -- the loop

These gates catch different failures; none replaces another:

- **Independent pre-push review** catches semantic and design defects before they consume a CI or
  hosted-review round.
- **CI** executes the complete test matrix and coverage checks.
- **Hosted reviewers** inspect the pushed diff with different models, instructions, and context.

1. **Watch CI to a terminal state.** Don't idle. If it fails, diagnose: fix if the failure is
   yours; if it's a known flake or pre-existing on main, say so with evidence.
2. **Wait for hosted review on the current HEAD.** Capture the exact PR HEAD SHA after pushing.
   Wait for all applicable hosted-review checks to reach a terminal state, and require at least one
   repository-configured AI reviewer or check to identify or attest that exact SHA. A stale or
   unidentifiable review does not satisfy the gate; rerun it or wait for a current review. An
   inapplicable or skipped reviewer is not a failure, but does not satisfy the at-least-one-review
   gate.
3. **Triage every comment** (bots and humans alike). For each one:
   - **Valid:** fix it, then reply saying what changed, and react 👍.
   - **Invalid:** reply explaining concretely why (with code evidence), and react 👎.
   - Never silently ignore a comment, and never resolve a thread without a reply.
4. **Clear active review requests.** Enumerate every active `REQUEST_CHANGES` review; resolved
   threads do not clear its formal review state. A human request remains blocking until that human
   re-reviews or a human maintainer explicitly dismisses it. A bot request requires current-head
   remediation and a replacement verdict, or a documented workflow-defined safe fallback, before a
   maintainer dismisses it.
5. **Escalate real trade-offs, don't guess.** If a comment needs a maintainer decision (a design
   choice, an API trade-off, a behavioral default), leave a comment containing: the background,
   your reasoning, the decision that needs making, the trade-offs (pros/cons of each option), and
   your recommendation. Then **poll every 30 minutes for a reply** and continue when it lands.
6. Wait for every applicable current-HEAD check to reach an accepted terminal state; classify any
   documented skip explicitly. Repeat until CI is green, a hosted AI review identifies or attests
   the current HEAD, no applicable check is pending or failing, no active review request remains,
   and no comment is outstanding.
7. For a user-facing capability change that affects its paired capability `README.md` and
   `docs/<capability>.md` surfaces, run the `docs-parity-reviewer` required by
   `agent_docs/review-checklist.md`. Treat blocking findings as merge blockers. Substantive changes
   restart the applicable review and verification gates.

## Before handing the PR back

Run this final metadata check after CI and comments have settled:

1. Capture the exact current title and body before dispatching the reviewer.
2. Dispatch a fresh no-history subagent from the stable policy-base checkout that has not worked on
   the PR.
3. Give it the PR URL, linked issue, current `base...HEAD` diff, final test status, and captured
   metadata.
4. Ask it to check only the title and body against this section and the root `AGENTS.md`.
5. Require either `current` or an exact replacement title and body. The reviewer returns text only;
   the implementing agent applies only objective rule corrections.
6. Before a metadata edit, record the timestamp. Metadata-only changes skip code review and CI but
   must wait for applicable `edited`-event checks created after that timestamp to reach their
   accepted terminal outcomes; stale checks on the same HEAD are not evidence. Triage any feedback.
   Code changes restart the full lifecycle.
7. After one replacement and its checks, repeat with another fresh subagent. Escalate conflicting,
   repeated, or discretionary rewrites to the maintainer instead of applying another replacement.
8. Immediately before handoff, re-read the title and body and compare them with the reviewed
   snapshot. Any difference restarts this metadata gate.
9. Hand the PR back only after the check reports `current` or the maintainer resolves the escalation.
10. Report the human-only AI-code checkbox separately.
