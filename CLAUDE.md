# Claude Agent Guidelines

## Project Overview

Taqsim is a water system simulation framework using event sourcing. It models water systems as directed acyclic graphs (DAGs) of nodes connected by edges, where water flows from sources through the network—getting stored, consumed, split—and eventually reaching sinks. All operations emit events, providing a complete audit trail of the simulation.

## Task Management Principles

### Avoid Task Jags

**Critical**: Avoid task jags at all cost. Jags are semantic changes in task direction:

- Going from implementing A to testing A
- Switching from implementing A to implementing B
- Any mid-stream change in the core task focus

Stay focused on the current task until completion. Delegate tasks to sub agents aggressively (3+ agents at a time), and remain at a higher level of abstraction and coordination, resisting the temptation of jumping in yourself for quick fixes.

### Delegation Strategy

**Always delegate orthogonal tasks to sub-agents**. Use the most appropriate agent from `agents/` for each task:

- Break down complex work into focused sub-tasks
- Route each sub-task to the specialist agent best suited for it
- Maintain clear task boundaries between agents
- Give agents all the context they need, and instruction on where to gather more context if needed.

## Context Awareness

### Library Implementation Details

`.context` contains git submodules of libraries used. Agents are **highly encouraged** to grep for implementation details of the files they work with to ensure consistency with library conventions.

### Documentation as Package Data

taqsim ships its `documentation/` directory inside the package. Downstream repos access docs via `taqsim.get_docs_path()` and sync them locally with a Makefile target. See `documentation/30_consuming_docs.md` for the full integration guide.

## Use Skills and Ask Questions

Use all skills that make semantic sense for the task.
Ask clarifying questions often to fill gaps. Better to clarify upfront than to implement the wrong solution.

---

## Python Package Management with `uv`

- **Use `uv` exclusively** for Python package management.
- Do **not** use `pip`, `pip-tools`, `poetry`, or `conda` directly.
- Commands:
  - Install: `uv add <package>`
  - Remove: `uv remove <package>`
  - Sync lockfile: `uv sync`
- Running:
  - Python scripts: `uv run <script>.py`
  - Tools: `uv run pytest`, `uv run ruff`
  - REPL: `uv run python`

---

## Ad-hoc Analyses and One-Time Scripts

- **Use shell heredoc syntax** for one-time data analyses and exploratory work.
- Do **not** create throwaway `.py` files or use alternative shell tools (awk, sed, etc.) for data manipulation.
- Python is more readable, maintainable, and powerful for these tasks.

**Preferred pattern:**

```bash
uv run python3 << 'EOF'
import pandas as pd

# Your analysis code here
df = pd.read_csv('data.csv')
print(df.describe())
EOF
```

**Why this matters:**

- **No file clutter** — no orphaned `temp.py` or `test_script.py` files
- **Self-documenting** — the command and its context live together in shell history or docs
- **Efficient** — Claude can generate complete, working analyses inline
- **Reproducible** — easy to copy-paste entire commands

This approach is **mandatory** for:

- Quick data inspections
- One-time transformations
- Exploratory analyses
- Data quality checks

For **reusable** logic that runs regularly, create proper Python scripts or modules.

---

## Python Coding Style

### Type Hints (mandatory)

- Always annotate function parameters and return types.
- Use built-in generics (`list`, `dict`, `tuple`, `set`) — **never** import `List`, `Dict`, etc. from `typing`.
- Use `|` for unions (Python 3.10+).
- Annotate variables where type is not obvious.

```python
def process_data(items: list[str]) -> dict[str, int]:
    ...

value: str | None = None
```

### Error Handling

- **Never** use bare `except`.
- Always raise meaningful errors with context.
- Prefer explicit error classes over generic `Exception`.

### Logging

- Use `logging` — never `print` — for runtime diagnostics.

### Formatting & Linting

- Use `ruff` for both linting and formatting:
  - Format: `uv run ruff format`
  - Lint + fix: `uv run ruff check --fix`

---

## Code Quality Standards

### High Signal-to-Noise Ratio

Strive for high signal-to-noise ratio in code:

- Clear, purposeful implementations
- Direct, readable solutions
- Declarative over imperative styles

### Structural Preferences

- **Avoid nested loops**: Prefer flat, pipeline-style code
- **Avoid deep nesting**: Keep nesting shallow (max 2-3 levels)
- **Prefer comprehensions and generators**: Use list/dict comprehensions for transformations
- **Use dataclasses and Protocols**: Leverage structural typing for clean interfaces

### Example Pipeline Style

```python
from functools import reduce

result = reduce(
    combine,
    filter(predicate, map(transform, data))
)

# Or with comprehensions
result = [transform(x) for x in data if predicate(x)]
```

### Example Pattern Matching (Python 3.10+)

```python
match command:
    case {"action": "create", "name": name}:
        return create_resource(name)
    case {"action": "delete", "id": id}:
        return delete_resource(id)
    case _:
        raise ValueError(f"Unknown command: {command}")
```

---

## Testability Requirements

### Control Side Effects via Dependency Injection

**CRITICAL**: Never use `datetime.now()` or `random.random()` directly in business logic. Always inject dependencies:

```python
# ❌ WRONG - untestable
def create_record():
    return {"created_at": datetime.now(), "id": random.randint(1, 1000)}

# ✅ CORRECT - testable
def create_record(clock: Callable[[], datetime], rng: random.Random) -> dict:
    return {"created_at": clock(), "id": rng.randint(1, 1000)}
```

**Why this matters**: Direct calls to `datetime.now()` and `random` are impure and non-deterministic, making tests flaky. Dependency injection allows:

- Controlled time in tests via fake clocks
- Deterministic random values via seeded RNGs
- Proper composition and testing

---

## Testing Philosophy

Good tests do not just check code; they shape its design. Tests are **contracts**: they describe what must stay true even if the implementation changes.

### Golden Rules

1. **Test behavior, not implementation**
   - Assert on outputs and public APIs.
   - Do not inspect private attributes like `_steps` unless no public API exists. If needed, add a public `.spec()` for testability.

2. **Each test should fail for one reason**
   - Keep assertions focused. Split broad tests into smaller ones.

3. **Prefer fast, deterministic tests**
   - No `sleep()`; control time with libraries like `freezegun` or dependency injection.
   - Control randomness by seeding or injecting RNGs.

4. **Use fakes over mocks**
   - Fake implementations are easier to read and maintain than heavy mocking.
   - Mock only at external boundaries (HTTP, file I/O, external services).

5. **Structure tests for readability**
   - Setup (Arrange) → Action → Assertion.
   - Use fixtures for repeated setup, but don't hide complexity in `conftest.py`.

### Test Coverage

- Use `pytest-cov` to measure coverage.
- Run with coverage: `uv run pytest --cov=src/<package> --cov-report=term-missing tests/`
- Coverage should be **used to find gaps**, not chased to 100%. A brittle 100% is worse than 85% meaningful coverage.

### Testing Conventions

#### File & Class Organization

- **One test file per module**: `test_<module>.py`
- **One test class per function/class under test**: `Test<ThingUnderTest>`
- Test methods: descriptive, snake_case, explain the behavior.
  Example: `test_fails_with_empty_dataframe`, not `test1`.

#### Categories of Tests

1. **Basic functionality**: happy paths with simple inputs.
2. **Error handling**: invalid inputs should raise the right exception with the right message.
3. **Edge cases**: empty data, null values, large inputs, unexpected types.
4. **Data preservation**: non-transformed fields, schema, and order remain intact.
5. **Integration paths**: small number of tests where the real pipeline runs end-to-end.

#### Assert Patterns

- Prefer **direct comparisons** for clarity.
- For complex structures, use `.to_dict()` or `.spec()` for clarity.
- Check types explicitly when relevant.

#### Error Testing

Always assert both **exception type** and **message fragment**:

```python
with pytest.raises(ValueError, match="no steps"):
    builder.build()
```

#### Fixtures

- Use fixtures sparingly and descriptively (`simple_df`, `df_with_missing_values`).
- Avoid fixture over-engineering; clarity > DRY.

---

## Feature Implementation Workflow

**CRITICAL**: Every feature implementation MUST follow this exact workflow. No shortcuts.

### Phase 1: Requirements Clarification

1. **Ask clarifying questions** before any design work
   - Understand the full scope
   - Identify edge cases
   - Clarify ambiguous requirements
   - Better to over-ask than under-deliver

### Phase 2: Signature & Contract Design

2. **Design Signatures & Type Contracts for EVERY component**
   - Use `Protocol` classes for service contracts (behavioral typing)
   - Use `dataclass` or `TypedDict` for data contracts
   - Use `Enum` for state/options definitions
   - **Focus on input/output types and method signatures** - no implementation details yet

3. **Verify Contracts with user**
   - Present all signatures/protocols for review
   - Iterate until the API surface is locked
   - Get explicit approval before proceeding
   - **No moving forward until the contract is finalized**

### Phase 3: Behavioral Specification

4. **Write test structure as behavioral contract**
   - Create test classes and methods with clear names matching the intended behavior
   - **Bodies contain only `pass`** - just the scaffold
   - Tests define the expected behavioral contract
   - Get verification that test cases cover requirements

### Phase 4: Test Implementation

5. **Write tests using Contracts only**
   - Use fakes/stubs based on the `Protocol` definitions
   - Tests must work against the defined signatures
   - **No real implementation exists yet**
   - Tests enforce the contract that implementations must fulfill

---

**⚠️ CHECKPOINT: Up until now, NOT A SINGLE LINE of real logic has been written.**

---

### Phase 5: Parallel Implementation

6. **Implement all test bodies** (extreme parallelism)
   - Spawn 10+ agents simultaneously
   - Each agent owns a specific test file/module
   - Tests remain contract-bound

7. **Implement all logic** (extreme parallelism)
   - Spawn 10+ agents simultaneously
   - Each agent owns a specific implementation
   - Run until all tests pass
   - **NO signature/contract changes allowed** unless:
     - Explicitly requested to user
     - Motivation for change is provided
     - User approves the change

### Workflow Enforcement

- Agents MUST refuse to skip phases
- Each phase requires explicit completion before next
- Signature/Contract changes after Phase 3 require user approval with justification
- Parallelism is MANDATORY in Phases 6-7, not optional

---

## Anti-Patterns (Avoid These)

- Asserting on private attributes (`._steps`, `._internal_state`).
- Overly specific error message checks (brittle wording).
- Giant integration tests covering all cases — push most variation down into unit tests.
- 100s of trivial tests (getter/setter, boilerplate) — test behaviors that matter.
- Hiding critical setup in nested fixtures.
- Using bare `except:` clauses.
- Importing deprecated typing generics (`List`, `Dict`, `Optional`).

**Tests should describe contracts, not internals.**
If your test breaks after a refactor that doesn't change behavior, the test was wrong.

---

## Documentation Standards

### Minimal Documentation During Prototyping

**CRITICAL**: Forego excessive docstrings unless specifically asked to.

- **NO lengthy docstrings** - They add significant context overhead
- **NO detailed comments** for self-explanatory code
- Focus on clean, self-documenting implementations
- Add documentation only when:
  - Explicitly requested by the user
  - Code is ready for production/publishing
  - Public API requires clarification

**Rationale**: During prototyping and development, verbose documentation significantly bloats context. Write clear, readable code first. Documentation can be added later when actually needed.
