---
title: Structured Output
description: Return a JSON Schema-shaped value the model is made to satisfy, with retries on violations.
---

# Structured Output

Give an agent a return value a program can consume: a JSON Schema goes to the model, and the reply
is checked against it before the run ends. Use it when the caller is a CI job, a workflow script, or
a parent agent -- anything that would otherwise parse prose. When the model replies with the wrong
shape, it is told exactly what was wrong and asked again.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

The example below uses an Anthropic model, so install the matching extra.

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SchemaOutput

REPORT = {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['pass', 'fail']},
        'findings': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['verdict'],
    'additionalProperties': False,
}

agent = Agent('anthropic:claude-fable-5', output_type=SchemaOutput(REPORT))
result = agent.run_sync('Review the diff.')
print(result.output)
#> ...
```

`result.output` is a `dict` that satisfies `REPORT`, or the run raised.

## What this adds over `StructuredDict`

Core's [`StructuredDict`](/ai/core-concepts/output/#structured-dict) already puts a schema on the
wire as an output tool. It does not check the reply: it compiles to `dict[str, Any]`, and the core
docs say so -- "No validation occurs; the model must correctly interpret the schema." A model that
invents a key or returns a value outside an `enum` is believed.

`SchemaOutput` wraps that same mechanism in an output function that validates and raises
`ModelRetry`, which core turns into a retry carrying the violations. The schema still reaches the
model unwrapped -- no parameter name leaks into it.

Every violation is reported at once, so one retry can fix all of them:

```text
Output does not match required schema: $.totally: additional property not allowed;
$.verdict: 'NOT-IN-ENUM' is not one of the allowed values (allowedValues: pass, fail)
```

A message names at most 50 violations and counts the rest, and is cut at 10,000 characters however
the schema nests. A systematic mistake across a long array needs one example, not a line per
element, and the message re-enters the model's context on every retry. An `anyOf` failure carries
each branch's own reasons, under the same cap, so the model can see which branch it was one field
away from satisfying.

## Enforced keywords

`type`, `properties`, `required`, `additionalProperties`, `items`, `enum`, `const`, `anyOf`.

`$schema`, `title`, `description`, `default` and `examples` are carried to the model and not checked. Non-recursive
`$ref`/`$defs` work: core inlines them, and validation runs against the inlined copy. Recursive
ones raise. A bare `true` or `false` may stand wherever a subschema may, as draft-06 and later
allow: `true` admits any value and `false` none, so `"items": false` permits only the empty array.

Everything else -- `minLength`, `pattern`, `format`, `minimum`, `maxItems`, `oneOf`, `allOf`,
`not`, `patternProperties`, tuple-form `items` -- reaches the model as documentation but is **not**
enforced. A `UserWarning` at construction names each one and where it appears. The
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/structured-outputs#output-format-configuration)
documents `format` as an unenforced annotation too; the numeric and length bounds are a real gap.
When a constraint has to hold, put it in the field's `description` as well, where the model reads it.

Two consequences of that follow draft-07 rather than being conveniences:

- `integer` accepts any number with a zero fractional part, so a model replying `1.0` to an
  `integer` field is correct and is not retried.
- `patternProperties` exempts the names its patterns match from `additionalProperties`, as
  draft-07 requires, whether that is `false` or a schema. The patterns decide membership, so a
  name matching none of them is still rejected; the subschemas they point at remain unenforced.
  Patterns run on the linear-time regex engine pydantic uses for its own `pattern` constraint, so
  a pattern such as `^(a+)+$` cannot stall the run on a crafted name. That engine has no lookaround
  or backreferences, and its character classes are Unicode-aware (`\d` matches any digit, as in
  Python's `re`, not only `0-9` as in ECMAScript). A pattern it rejects declares nothing and so
  exempts nothing.

A provider may rewrite the schema on its way to the model -- core runs each provider's
`json_schema_transformer` over output tools -- while validation still enforces the schema you
passed. That is deliberate, since the contract is yours, but a provider that drops a keyword can
cost a retry.

## Unsatisfiable schemas are refused early

A schema that admits no value is caught when `SchemaOutput` is built, not after a run has spent its
retry budget failing. A finding on a path every valid instance must have (the root, or a chain of
required properties) raises `UserError`; one on an optional path -- an unrequired property, an array
element, one `anyOf` branch -- warns instead, because the run can still succeed.

```python
from pydantic_ai_harness import SchemaOutput

SchemaOutput({'type': 'object', 'properties': {}, 'required': ['verdict'], 'additionalProperties': False})
#> UserError: ... $.verdict: is required but is not declared in properties, ...
```

Checks: a root that is not an object, a required property that `additionalProperties: false`
forbids or whose schema is `false`, an `enum` that is empty or whose every value fails the rest of
its node (its `type`, `required`, `properties` and so on), a `const` that fails likewise or sits
outside its `enum`, an `anyOf` none of whose branches can be satisfied, and crossed bounds
(`minimum` above `maximum`, an exclusive bound meeting its partner, and the
`length`/`items`/`properties` pairs). Crossed bounds are linted even though they are not enforced
at validation time: an impossible bound is a bug in the schema either way, and construction is the
cheap place to say so.

A keyword is held against a node only where it binds. Crossed numeric bounds on a node typed
`string` admit every string, and `required` on a node that does not declare `type: 'object'` leaves
a non-object value acceptable; the lint treats both as satisfiable, which is what validation does
with them. An `anyOf` branch is ruled out only by a finding on its own required path, not by one
under an optional property or array element it can leave unset.

## Retries

`max_retries` defaults to 3 and forwards to `ToolOutput.max_retries`, so it is core's output budget
rather than a second counter. Core's own default is 1, which is tight for a shape the model has to
get right first time. Exhausting it raises `UnexpectedModelBehavior`.

The output tool is named `structured_output` and described as "Return your final response in the
requested structured format" unless you pass `name` or `description`.

## Keep schemas at module scope

A schema rebuilt per call produces a per-call tool definition, which moves the prompt prefix and
stops a fan-out sharing a prompt cache. Define it once at module scope and reuse it.

## Telemetry

`SchemaOutput` emits no spans. Core already spans the model request and the output tool call, so a
span per validation would repeat the run span without recording a decision. The lint and the
unsupported-keyword warning happen at construction, outside any `RunContext`, where there is no
tracer to write to.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/structured_output/).

## API reference

::: pydantic_ai_harness.structured_output.SchemaOutput
