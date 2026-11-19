# Engineering Decisions

Each entry records a decision, its rationale, and the phase it was made in.

## D-001 — New repository location (Phase 00)
The workspace was not a Git repository and contained no prior fraud-risk work.
A new repository was created with `git init -b main`
and a single empty root commit (`chore: initialize repository`) so phase branches have a common,
reviewable base. `main` receives no further commits; `dev` was not created.

## D-002 — Branch chaining (Phase 00)
Branch slugs follow `manifest.json` (e.g. `phase/02-data-validation`, `phase/04-baseline-model`).
Phase `NN` implementation branches are created from the previous phase's **test** branch
`phase/test-(NN-1)-slug`, so each phase builds on reviewed implementation *and* its tests and the
final branch contains the whole project. Each test branch is created from its own implementation
branch, as required.

## D-003 — Execution-log commits (Phase 00)
The orchestration prompt requires `docs/execution/*` to be updated after every phase, including
test results and commit hashes that only exist after the test commit. These updates are therefore
made in a separate `docs(execution): ...` commit on the test branch, after the test-only commit.
They touch only `docs/execution/` — never `src/`, configuration, or dependencies — so the test-only
commit itself stays test-only.

## D-004 — No AI co-author trailers (Phase 00)
The user's specification explicitly forbids AI-agent co-author trailers; commits use the existing
local Git identity unchanged.

## D-005 — Isolated virtual environment (Phase 00)
All commands run inside a Git-ignored `.venv` (Python 3.13.0 on Windows 11) so the global
interpreter is untouched. `requires-python = ">=3.11"`; CI tests 3.11 and 3.13.

## D-006 — Strict typing scope (Phase 00)
`mypy --strict` covers `src/` (the documented `mypy src` gate). Tests are linted and formatted by
Ruff but not type-checked strictly.
