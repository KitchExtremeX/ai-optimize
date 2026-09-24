# Optimized prompt

You complete one task. Do not expand the scope.

## Task
make a marketing email

## Scope
The original request is short and leaves out the reader, the output shape, or the inputs. Those slots are explicit below. Do not fill them with guesses.

## Goal
Get a reply from a busy cafe owner

## Audience
Independent cafe owners who already buy beans wholesale

## Constraints
Under 120 words. No discount. One clear ask.

## Inputs
Use only material the user includes with this prompt. If the task needs source text, code, data, or a link and it is not there, list what is missing and stop.

## Output
Produce the message text only. Do not add a brand strategy, a subject-line list, or a send plan unless the task asked for them.

Structure the answer so a reviewer can score it without asking what was meant.
If two instructions conflict, name the conflict and follow Constraints.
When you had to assume something non-obvious, add a final line that starts with "Assumptions:". Otherwise omit that line.

## Original request
> make a marketing email

# Eval notes

## What good looks like

- A reviewer can tell, from the answer alone, that it responds to: make a marketing email
- The answer has a visible structure and does not add a second deliverable.
- Missing source material is listed. It is not replaced with invented facts.
- The answer follows the output rule (message).
- The answer serves this goal: Get a reply from a busy cafe owner
- The language fits this audience: Independent cafe owners who already buy beans wholesale

## Edge cases

- The user sends the prompt with no source document, code, or data attached.
- The source material contradicts the constraints.
- The request can be read as either a short reply or a structured artifact.
- The user includes only part of the material and the rest would have to be guessed.

## Scoring checks

- **Task preserved** — pass if: the answer is clearly about: make a marketing email
- **Checkable shape** — pass if: a reviewer can score the answer without asking what the prompt wanted
- **Missing inputs** — pass if: when no source material is attached, the answer lists what it needs and does not invent it
- **Output rule** — pass if: the answer matches the message rule: Produce the message text only. Do not add a brand strategy, a subject-line list, or a send plan unless the task asked for them.
- **Single deliverable** — pass if: the answer does not add a second task the original request did not ask for
- **Constraints** — pass if: the answer respects: Under 120 words. No discount. One clear ask.
- **Audience** — pass if: the wording fits: Independent cafe owners who already buy beans wholesale

## What changed

- Split the request into task, goal, audience, constraints, inputs, and output.
- Told the answering model to stop when required inputs are missing.
- Marked the original request as underspecified.
- Placed the given goal into the prompt.
- Placed the given audience into the prompt.
- Placed the given constraints into the prompt.
- Selected the local 'message' output rule from the wording. This is a template, not a model rewrite.
- Added eval notes: what good looks like, edge cases, and pass/fail checks.

---
mode: dry-run
model: (none — local template, no API call)
