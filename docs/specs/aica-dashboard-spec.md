# AICA Dashboard & Webhook Server Spec

## Context

### What is pydantic-harness?

[pydantic-harness](https://github.com/pydantic/pydantic-harness) is a Python package that publishes pre-made capabilities (AbstractCapability subclasses) for [Pydantic AI](https://github.com/pydantic/pydantic-ai), a production-grade agents framework. It exists so the Pydantic AI team can iterate fast on composite capabilities without the strict review standards of pydantic-ai core, and so external users can build and publish their own capability packages.

### How development works

Development is AICA-driven (AI Code Assistant). The workflow is label-based on GitHub:

1. **Issue opened** -- describes a capability to build or a bug to fix
2. **`aica:write-plan`** label -- AICA opens a PR with `PLAN.md` only
3. **Human review** -- maintainers review the plan, leave comments
4. **`aica:update-plan`** label -- AICA researches questions and updates the plan
5. **`aica:implement-plan`** label -- AICA implements the approved plan
6. **Ralph loop** -- automated feedback loop handles PR review comments iteratively (TRIAGE -> GOALS -> PLAN -> CODE -> VERIFY -> REVIEW -> PUBLISH -> WAIT)
7. **`aica:research`** label -- AICA answers questions on issues with linked sources
8. **`aica:review-plan`** label -- AICA reviews a plan PR

### Why not GitHub CI?

AICAs run as external processes, not in GitHub Actions. Running AI agents in CI creates security vulnerabilities: malicious PRs could inject prompts, exfiltrate secrets, or trigger unwanted actions. The external webhook model keeps the AI execution separate from the CI trust boundary.

### The ralph loop

The ralph loop is a state machine that processes PR review feedback:

- **Phases**: TRIAGE, GOALS, PLAN (with review/research sub-loop), CODE, VERIFY, REVIEW, PUBLISH, WAIT, DONE
- **DDD+ protocol**: classifies review comments as do/dismiss/discuss/waiting/done
- **BLOCKED_QUESTIONS**: when the AICA needs human input, it writes `questions.json` and blocks until answered
- **Moderate autonomy**: auto-resolves addressed threads, asks for confirmation on dismissals
- **State file**: `ralph-state.json` tracks current phase, iteration, history

### Maintainers

- DouweM, samuelcolvin, dmontagu, dsfaccini, adtyavrdhn
- Only pydantic team members can sign up and access the dashboard (auth via existing auth worker binding)

### Content moderation

- `slop:suspected` label -- auto-tagged by a bot on new issues/PRs suspected of being AI-generated slop
- `slop:confirmed` label -- confirmed slop, auto-closed
- PRs modifying `pyproject.toml` or `uv.lock` from non-team members are auto-closed

---

## Stack

- **Runtime**: Cloudflare Workers
- **API layer**: [Hono](https://hono.dev/) -- lightweight, replaces worker boilerplate
- **Frontend**: React + [TanStack Query](https://tanstack.com/query)
- **Auth**: existing auth worker binding (service binding) -- only pydantic members can sign up/access
- **Data**: GitHub API is the source of truth. No separate database unless caching demands it

---

## Design principles

1. **GitHub is the source of truth** -- the dashboard reads from GitHub API, writes via labels and comments. Never store state that contradicts GitHub
2. **Dashboard supplements GitHub** -- only build UI for things that are uncomfortable or limited in the GitHub UI
3. **Label-driven** -- AICA actions are triggered by labels on GitHub. The dashboard provides a convenient way to apply labels, but the labels are the trigger
4. **Delegate UI/UX research** -- the implementing agent should research best practices for dashboard UI/UX. This spec defines the features, not the visual design

---

## Core features

### 1. Overview dashboard

**Purpose**: give maintainers a single view of all active work across the repo.

**Data to surface**:
- All open issues with labels, priority, assignee, and AICA status
- All open PRs with labels, CI status, review status, and ralph loop phase
- Progress indicator per PR: which ralph loop phase it's in, how many iterations
- Last activity timestamp per issue/PR

**Why GitHub UI isn't enough**: GitHub doesn't show ralph loop phase, AICA status, or aggregate progress. You have to open each PR individually to understand where things stand.

### 2. Question surfacing

**Purpose**: when the AICA hits BLOCKED_QUESTIONS, surface the questions prominently so maintainers can answer quickly.

**Behavior**:
- Monitor for `questions.json` state (via webhook events or polling ralph-state.json)
- Display questions with the original context (which PR, which phase, what the AICA is trying to do)
- Allow answering directly from the dashboard (writes to `questions.json` or posts a GitHub comment that the AICA can parse)
- Highlight unanswered questions with urgency -- these block AICA progress

**Why GitHub UI isn't enough**: questions are buried in PR comments or in a JSON file in the repo. There's no notification system that aggregates "all blocked AICAs" across PRs.

### 3. AICA trigger controls

**Purpose**: apply AICA labels from the dashboard without navigating to GitHub.

**Behavior**:
- Per-issue: button to apply `aica:research`, `aica:write-plan`
- Per-PR: button to apply `aica:update-plan`, `aica:implement-plan`, `aica:review-plan`
- Confirmation dialog before applying (accidental triggers are expensive)
- Show current labels to avoid duplicate application

**Why GitHub UI isn't enough**: applying labels in GitHub requires opening the issue/PR, finding the label dropdown, searching for the right label. The dashboard makes this one click.

### 4. Slop detection feed

**Purpose**: quickly review and act on suspected AI slop.

**Behavior**:
- List all issues/PRs with `slop:suspected` label
- Show the content preview (title, first paragraph of body)
- One-click actions: confirm as slop (apply `slop:confirmed`, auto-closes), dismiss (remove label), or open in GitHub for closer inspection

**Why GitHub UI isn't enough**: there's no "slop queue" view in GitHub. You'd have to filter by label and open each one.

### 5. Activity timeline

**Purpose**: see recent AICA actions and phase transitions at a glance.

**Data to surface**:
- Label application events (who applied what label, when)
- Ralph loop phase transitions (PR #X moved from CODE to VERIFY)
- Human interventions (comments, reviews, question answers)
- AICA actions (plan created, code pushed, threads resolved)

**Source**: GitHub webhook events + ralph-state.json phase_history

**Why GitHub UI isn't enough**: GitHub's activity feed is per-repo and mixes all event types. The dashboard filters to AICA-relevant events only.

---

## Webhook integration

### Endpoint

`POST /webhook/github` -- receives GitHub webhook events.

### Events to handle

| Event | Action |
|-------|--------|
| `issues.labeled` | If label matches `aica:*`: dispatch AICA action (spawn claude process, manage worktree, run loop) |
| `pull_request.labeled` | Same as above for PR labels |
| `issue_comment.created` | Check if AICA is in BLOCKED_QUESTIONS on related PR; if comment contains answer format, update `questions.json` and resume loop |
| `pull_request_review.submitted` | Trigger re-triage if ralph loop is in WAIT or DONE for the PR |
| `issues.opened` | Run slop detection (AI classifier), apply `slop:suspected` if detected |
| `pull_request.opened` | Same as above; also check if PR modifies dependencies from non-team member |

### AICA dispatch

When a label triggers an AICA action, the webhook server needs to:

1. Determine the target (issue number or PR number)
2. Determine the action (research, write-plan, update-plan, implement-plan, review-plan)
3. Spawn a claude process in the appropriate context (worktree for PRs, fresh checkout for issues)
4. Track the process (PID, status, start time)
5. Remove the triggering label after the action completes

**Process management considerations**:
- The webhook server runs on Cloudflare Workers, which are stateless and short-lived
- Long-running AICA processes need to run on a separate machine (e.g., a VM or dedicated server)
- The webhook server should send a message to a queue or invoke an API on the AICA runner, not run the process itself
- The runner reports status back via GitHub labels/comments and by updating ralph-state.json

### Webhook secret

GitHub webhook payloads must be verified using the webhook secret (HMAC-SHA256). Reject any request that fails verification.

---

## API routes (Hono)

```
GET  /api/overview          -- aggregated view of all issues/PRs with status
GET  /api/questions          -- all currently blocked AICAs with their questions
POST /api/questions/:id      -- answer a question (writes to GitHub)
POST /api/trigger/:target    -- apply an AICA label to an issue/PR
GET  /api/slop               -- suspected slop queue
POST /api/slop/:id/confirm   -- confirm slop (close issue/PR)
POST /api/slop/:id/dismiss   -- dismiss slop suspicion (remove label)
GET  /api/activity           -- recent AICA activity timeline
POST /webhook/github         -- GitHub webhook endpoint
```

---

## Auth

- Use existing auth worker binding (service binding in Cloudflare)
- Only pydantic team members can sign up and access
- All API routes require authentication
- The webhook endpoint uses the GitHub webhook secret for verification (not user auth)

---

## Open questions for implementing agent

These are decisions the implementing agent should research and decide:

1. **AICA runner architecture**: how should the webhook server communicate with the machine(s) running claude processes? Options: queue (SQS, Cloudflare Queue), direct API call, SSH, etc.
2. **State caching**: should the dashboard cache GitHub API responses in KV/D1, or always fetch fresh? Consider rate limits vs staleness
3. **Real-time updates**: WebSocket, SSE, or polling for dashboard updates?
4. **Slop classifier**: what model/approach for detecting AI-generated slop? Fine-tuned classifier, heuristics, or API call to Claude?
5. **UI framework**: any specific React component library or design system? (e.g., shadcn/ui, Radix)
