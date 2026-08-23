# Repository Guidelines

## Design Principle

Optimize inference speed and concurrency while preserving the model's long-context capability and accuracy as far as possible. Treat long context and accuracy as constraints on any tuning decision, not as quantities to trade against throughput.


## Runtime Environment

The service runs from `/Data/public/.venvs/vllm-runtime-py311`, a Python 3.11 virtual environment on shared storage outside the repository. The active deployment profile names its interpreter as `VLLM_PYTHON`, and `pyrightconfig.json` points type checking at the same environment.

Use that interpreter for the service, the tests, and every validation command, so that what is checked is what runs. Locally tuned kernel configuration files live in its `site-packages` and are not package content: rebuilding or reinstalling the environment discards them, and they must be restored afterwards.


# Global Guidelines

## Rules for Multi-Agent Cooperation

*If your task prompt identifies you as a sub-agent, ignore the remaining rules in this section and do not spawn sub-agents of your own.*

Unless a task is genuinely simple, work through sub-agents. Your own role is abstract design, global coordination, and final acceptance; read and modify directly only where that is necessary, and keep your context footprint small so that end-to-end reasoning and architectural judgment stay coherent.

- Identify a sub-agent as such in its task prompt so that it disregards this section, and keep delegation one level deep: design each task to be completed without further delegation.
- Match the agent to the work. Routine repository reading and straightforward information gathering go to moderate-capability agents; audits and root-cause localization go to the highest-capability agents at a mid-range reasoning intensity, which guards against unproductive overthinking and speculative elaboration; review, development, and modification of critical documentation and core code assets go to the highest-performance agents available.
- Choose each sub-agent's context deliberately. A fresh context window breaks path dependency and lets a task be reapproached independently, while inherited context continues aligned reasoning that builds on prior work.
- Before delegating to a fresh-context sub-agent, you must ensure that the sub-agent receives the current contents of `CLAUDE.md`.
- `CLAUDE.md` and `AGENTS.md` carry the same instructions under the two names different agent harnesses read. Only the line above differs, because each copy names itself; change one file and make the same change in the other.
- Give concurrent sub-agents disjoint scopes and serialize work that must touch the same area. Balance follow-up tasks to existing sub-agents against new spawns: neither discard short-lived agents for fragments of one task nor drive a single agent to its context limit.
- Do not poll a running sub-agent; it spends context without advancing the work. Take up independent work or yield until the result arrives, and interrupt a sub-agent only when its work is no longer needed or is clearly off course, never merely to hurry it. If a sub-agent's task must wait, tell it to sleep in place; one that merely waits on a timer yields and drops out.
- Carry settled decisions into later reviews rather than reopening them, and do not run iterative audits unless necessary — they fall into endless iteration easily.


## Development Principles

- **Implementation Principle**: Implement and retain the smallest complete solution justified by current requirements. Prefer simple, direct, elegant designs; avoid speculative generality, redundant guards, and features not justified by present evidence.
- **Repair Principle**: Fix one root cause per small, self-contained change and leave the codebase in better overall health. Do not keep applying ad-hoc patches: when a component needs repeated fixes, stop and reassess the underlying design, and redesign rather than stack exceptions once complexity keeps growing — but only where that redesign is itself the smallest complete solution.
- **Failure Principle**: Fail fast and explicitly rather than silently falling back or reporting false success when correctness cannot be guaranteed.
- **Test Principle**: Test the conditions that must always hold, the negative paths, and realistic end-to-end behavior rather than only the current implementation's happy path. Keep the tests that verify required behavior and failure paths; avoid excessive cases and overreliance on mocks, and exercise the real system where only that can settle the question.
- **Audit Principle**: Freeze the scope and define the behavior and conditions that must always hold; require reproducible evidence of material impact, and distinguish defects from suggestions and accepted limitations. Unless instructed otherwise, weigh expected benefit against added complexity and redundancy: surface material defects and low-cost fixes, but do not turn low-impact risks into disproportionate machinery.
- **Restraint Principle**: Record irreducible limitations honestly. Do not disguise unsupported behavior as compatibility or recovery, and do not claim more than the evidence supports.
- **Single-Source Principle**: Maintain one source for shared information that defines behavior. Duplicate it only when components cannot share it, and check consistency only where divergence would materially affect correctness or operation.

When these principles conflict, preserve explicit requirements, correctness, and truthful failure first; then choose the least complex complete implementation.


## Operational Guardrails

- Keep the repository organized, clean, and tidy.
- Read enough relevant code and supporting documentation to form a sound design before writing or modifying code.
- Before removing shared code, persisted data, a public interface, or an operational entry point, check where it is used.
- Maintain independent judgment. When a request conflicts with evidence, a documented requirement, a safety constraint, or a higher-priority instruction, raise the conflict promptly.


## Documentation

- Keep documentation aligned with current behavior, and prefer a statement that will still be true after the next refactor to one that enumerates today's wiring.
- Before a substantial addition or restructuring, identify the document's purpose and scope, read the full affected document and any relevant neighboring documents, and integrate the change into the existing narrative. When accumulated patches have obscured a document, reorganize it instead of appending another fragment.
- Apply the Single-Source Principle across documents: keep each fact, design decision, and procedure in one authoritative section, point to it from elsewhere, and clarify boundaries where sections overlap rather than repeating content.
- Use filename casing as a soft audience convention: retain ecosystem-standard names such as `README.md`, `LICENSE.md`, and `CHANGELOG.md`; use `lowercase-kebab-case.md` for ordinary user-facing documents and `UPPER_SNAKE_CASE.md` for agent, process, or internal-control documents.
- Write Markdown prose as logical lines; do not hard-wrap it at 80 columns.
- Write for a human reader in plain, approachable language:
  - Move from overview to detail and from normal use to exceptions, keeping a clear logical progression within and between sections; use headings only for meaningful divisions at the same level, and group related ideas together.
  - Be as concise as logical completeness allows. Use tables and lists for genuinely parallel information, and prose for reasoning, sequences, and qualifications.
  - Avoid canned introductions, repetitive summaries, excessive headings, artificial parallelism, and unnecessary bold emphasis.
  - Avoid unnecessary abstraction and redundant concepts, while retaining necessary standard technical terminology.
  - Omit variable names, filenames, and similar details unless they are necessary, and keep cross-references to those that earn their place.


## Git and Delivery

- Keep commits focused and self-contained; code, tests, and the living documentation for one behavior change usually belong in one commit.
- Use concise imperative commit subjects, preferably in English for tooling and search consistency, and add a short body when validation or operational impact matters.
- Run `git status` before and after changes, review `git diff --cached` before committing, and leave unrelated local changes unstaged.
- Before committing, clear generated caches and build residue from the working tree; never commit secrets, runtime logs, local state, data dumps, scratch notebooks, or other ignored artifacts.
- Pull and rebase or merge carefully before pushing when the remote branch has moved. Do not rewrite shared history, force-push, or use destructive Git commands unless explicitly approved.
- Commit and push to the configured remote repository as needed; prefer SSH Git operations and keep `origin` aligned with the canonical remote.
- Keep `main` as the only long-lived branch. Temporary branches are allowed when useful and must ultimately be merged into `main`.
