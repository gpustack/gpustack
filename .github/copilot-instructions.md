# Copilot Code Review — GPUStack

GPUStack is an open-source GPU cluster manager for running AI models. The
single Python distribution provides server, worker, and CLI runtimes. The
server exposes FastAPI APIs, manages SQLModel/SQLAlchemy data, schedules work,
and coordinates controllers. Workers discover and manage local resources and
run inference backends. Higress fronts inference traffic as the AI gateway.

Read `CLAUDE.md` and `AGENTS.md` before reviewing. They are the source of
truth for repository conventions. Keep review feedback specific and actionable,
and cite the file and line. Report correctness, security, data-loss, and
regression risks; do not report stylistic preferences already enforced by
formatters and linters.

## Review priorities

- Treat every new or changed route as security-sensitive. Check its deliberate
  tier in `gpustack/routes/routes.py`: authenticated user, tenant-scoped user,
  platform administrator, or worker/cluster system principal. Flag missing or
  overly broad authentication and authorization dependencies.
- Treat every tenant-reachable query as security-sensitive. List queries must
  use `tenant_list_conditions`; single-resource access must use
  `assert_resource_visible` or an equivalent established ownership guard from
  `gpustack/api/tenant.py`. Never allow a missing ownership predicate to expose
  another organization’s data.
- Do not allow API keys, access tokens, passwords, SAML assertions, cloud
  credentials, or other secrets to be logged, returned, serialized into errors,
  or included in tests and documentation.
- A behavior change or bug fix needs a focused test under the matching `tests/`
  package. Async tests require `@pytest.mark.asyncio`; tests must not use a real
  database, network, or GPU.

## Repository conventions

- Python is formatted with Black (88 columns) and checked by Flake8. Preserve
  existing quote style; do not suggest unrelated formatting changes.
- Prefer simple, explicit code. Flag speculative abstractions, mutable shared
  state without synchronization, swallowed exceptions, and error paths that
  leave persisted or external state inconsistent.
- Public functions require type hints. Use `Optional` and `List` from `typing`
  where applicable. Comments explain behavior and rationale, never revision
  history.
- Generated clients in `gpustack/client/generated_*.py` come from
  `gpustack/codegen/` and must not be hand-edited. Review the templates or
  generator and require `make generate` as a reminder when they change; do not
  claim generated output is present or absent.
- Docs in `docs/` are English-only. A new documentation page must be added to
  `mkdocs.yml` navigation. `README.md` is canonical; meaningful changes there
  require noting that Chinese and Japanese translations need updating.
- Commit-message and PR-title conventions are not source-code defects. Do not
  post them as inline review findings.

## Database migrations

- Released migration revisions are immutable. Unreleased schema changes belong
  in the current release bundle unless ordering requires a standalone revision.
- Changes must support PostgreSQL, openGauss, MySQL, and OceanBase. Review for
  dialect assumptions, especially PostgreSQL enums/JSON functions and MySQL
  differences. `dialect.name` distinguishes only PostgreSQL and MySQL; use the
  existing migration utilities for openGauss-specific detection.
- Migrations must be re-runnable with `table_exists` and `column_exists` guards
  where appropriate, and must provide a real downgrade or explain why reversal
  is impossible. Ownership and permission migrations require special scrutiny.

## Out of scope

- `gpustack/client/generated_*.py` (generated; review its templates instead).
- Python bytecode and cache directories such as `__pycache__/`.

## Known non-findings

- Do not ask to reformat code solely for quote normalization: Black is
  configured to preserve existing quote style.
- Do not request network, database, or GPU integration tests: project tests
  intentionally mock those boundaries.
