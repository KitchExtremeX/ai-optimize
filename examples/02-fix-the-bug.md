# Optimized prompt

You complete one task. Do not expand the scope.

## Task
write code to fix the bug

## Scope
The original request is short and leaves out the reader, the output shape, or the inputs. Those slots are explicit below. Do not fill them with guesses.

## Goal
Not provided. Do not invent a goal. If the task cannot be finished without one, stop and ask.

## Audience
Not provided. Use plain language a busy builder can check. Do not invent a persona.

## Constraints
None provided. Stay concise. If a needed input is missing, name it and stop instead of guessing.

## Inputs
Use only material the user includes with this prompt. If the task needs source text, code, data, or a link and it is not there, list what is missing and stop.

## Output
Produce the code change or the diagnosis the task asks for. If the failing behavior, the code, or the expected result is missing, list those gaps and stop. Do not rewrite unrelated code.

Structure the answer so a reviewer can score it without asking what was meant.
If two instructions conflict, name the conflict and follow Constraints.
When you had to assume something non-obvious, add a final line that starts with "Assumptions:". Otherwise omit that line.

## Original request
> write code to fix the bug

# Eval notes

## What good looks like

- A reviewer can tell, from the answer alone, that it responds to: write code to fix the bug
- The answer has a visible structure and does not add a second deliverable.
- Missing source material is listed. It is not replaced with invented facts.
- The answer follows the output rule (code).
- No goal was provided, and the answer does not invent one.

## Edge cases

- The user sends the prompt with no source document, code, or data attached.
- The source material contradicts the constraints.
- The request can be read as either a short reply or a structured artifact.
- The user includes only part of the material and the rest would have to be guessed.

## Scoring checks

- **Task preserved** — pass if: the answer is clearly about: write code to fix the bug
- **Checkable shape** — pass if: a reviewer can score the answer without asking what the prompt wanted
- **Missing inputs** — pass if: when no source material is attached, the answer lists what it needs and does not invent it
- **Output rule** — pass if: the answer matches the code rule: Produce the code change or the diagnosis the task asks for. If the failing behavior, the code, or the expected result is missing, list those gaps and stop. Do not rewrite unrelated code.
- **Single deliverable** — pass if: the answer does not add a second task the original request did not ask for

## What changed

- Split the request into task, goal, audience, constraints, inputs, and output.
- Told the answering model to stop when required inputs are missing.
- Marked the original request as underspecified.
- Selected the local 'code' output rule from the wording. This is a template, not a model rewrite.
- Added eval notes: what good looks like, edge cases, and pass/fail checks.

---
mode: dry-run
model: (none — local template, no API call)
