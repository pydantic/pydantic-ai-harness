# Development Processes

## Software Factory Flow

How a capability goes from idea to merged code:

```mermaid
flowchart TB
    issue["Issue opened\n(capability request)"] --> label_plan["Label: aica:write-plan"]
    label_plan --> pr["AICA opens PR\nwith PLAN.md"]
    pr --> human_review["Human review\n(comments on plan)"]
    human_review -->|"needs changes"| label_update["Label: aica:update-plan"]
    label_update --> aica_research["AICA researches\nquestions with sources"]
    aica_research --> update_plan["AICA updates PLAN.md\n(decisions in PR comment)"]
    update_plan --> human_review
    human_review -->|"plan approved"| label_impl["Label: aica:implement-plan"]
    label_impl --> implement["AICA implements plan\n(code + tests)"]
    implement --> code_review["Code review\n(human + ralph loop)"]
    code_review -->|"feedback"| ralph["Ralph loop handles\nreview feedback"]
    ralph --> code_review
    code_review -->|"approved"| merge["Merge to main"]
    merge --> release["Release to PyPI\n(tag-triggered)"]
```

## Ralph Loop State Machine

The automated feedback loop that processes PR review comments:

```mermaid
stateDiagram-v2
    [*] --> TRIAGE

    TRIAGE --> BLOCKED_QUESTIONS : discuss items
    TRIAGE --> GOALS : actionable items
    TRIAGE --> DONE : nothing actionable

    GOALS --> PLAN : goals defined

    state "PLAN (read-only)" as PLAN
    PLAN --> PLAN_REVIEW

    state "Plan Review Sub-Loop (max 3)" as PlanSubLoop {
        PLAN_REVIEW --> PLAN_RESEARCH : researchable gaps
        PLAN_REVIEW --> CODE_READY : plan is solid
        PLAN_RESEARCH --> PLAN_REVIEW
    }

    CODE_READY --> CODE

    CODE --> VERIFY
    CODE --> PLAN : impossible instruction

    VERIFY --> REVIEW : tests pass
    VERIFY --> CODE : tests fail (max 3)
    VERIFY --> BLOCKED_QUESTIONS : 3 failures

    REVIEW --> CODE : blocking issues
    REVIEW --> PUBLISH : clean

    PUBLISH --> BLOCKED_QUESTIONS : confirmations
    BLOCKED_QUESTIONS --> PUBLISH : approved

    PUBLISH --> WAIT
    WAIT --> TRIAGE : 10min + conflict check

    BLOCKED_QUESTIONS --> TRIAGE : answers received
    DONE --> [*]
```

## Contribution Flow

For external contributors:

```mermaid
flowchart LR
    idea["Have an idea"] --> issue["Open issue\n(capability request)"]
    issue --> discuss["Discussion\nwith maintainers"]
    discuss -->|"approved"| fork["Fork / branch"]
    fork --> implement["Implement capability\n+ tests"]
    implement --> pr["Open PR\n(no dep changes!)"]
    pr --> review["Code review"]
    review -->|"feedback"| implement
    review -->|"approved"| merge["Merge"]
```

## Publishing a Capability Package

For building and publishing standalone capability packages:

```mermaid
flowchart TB
    start["Copy template/ directory"] --> rename["Rename package\n(pydantic-ai-*)"]
    rename --> implement["Implement\nAbstractCapability subclass"]
    implement --> hooks["Wire up hooks\n(before_run, wrap_model_request, etc.)"]
    hooks --> tests["Write tests\n(TestModel, no API calls)"]
    tests --> ci["CI green\n(lint + typecheck + test)"]
    ci --> publish["Publish to PyPI"]
    publish --> register["Users register via\ncustom_capability_types"]
```

## Human Review Types

What maintainers focus on during plan and code review:

```mermaid
flowchart TB
    comment["Reviewer comment"] --> classify{"Type?"}
    classify -->|"Missing research"| research["Ask for references\nthat back up claims"]
    classify -->|"Needs verification"| verify["Request test/script\nto verify behavior"]
    classify -->|"Design question"| discuss["Discussion needed\n(scope, approach)"]
    classify -->|"Quality issue"| fix["Direct fix request\n(code, tests, docs)"]

    research --> aica_research["Label: aica:research\nAICA answers with sources"]
    verify --> experiment["Write one-off script\nto validate decision"]
    discuss --> decision["Maintainer decides\nAICA implements"]
    fix --> implement["AICA implements\nvia ralph loop"]
```
