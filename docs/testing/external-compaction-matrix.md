# External Compaction Test Matrix

This matrix defines required coverage for external compaction flows across backend mode,
MCP mode, and write target.

## Matrix

| Backend | MCP | Write Target | Covered By |
|---|---|---|---|
| `local` | off | `runtime` | `tests/test_external_compaction_matrix.py` |
| `local` | on | `runtime` | `tests/test_external_compaction_matrix.py` |
| `local` | off | `templates` | `tests/test_external_compaction_matrix.py` |
| `local` | on | `templates` | `tests/test_external_compaction_matrix.py` |
| `shared_file` | off | `runtime` | `tests/test_external_compaction_matrix.py` |
| `shared_file` | on | `runtime` | `tests/test_external_compaction_matrix.py` |
| `shared_file` | off | `templates` | `tests/test_external_compaction_matrix.py` |
| `shared_file` | on | `templates` | `tests/test_external_compaction_matrix.py` |

## Required Companion Suites

- Schema and validation behavior: `tests/test_external_compaction.py`
- Write-scope policy hardening: `tests/test_external_compaction_write_guard.py`
- CLI end-to-end flows and automation contracts: `tests/test_cli.py`
- Command inventory parity: `tests/test_command_contract.py`

## Fault Injection Coverage

- Malformed payload rejection: `tests/test_cli.py::test_cli_external_compaction_apply_invalid_payload_json_mode`
- Template policy gate denial + safe override: `tests/test_cli.py::test_cli_external_compaction_template_write_target_requires_opt_in`
- Partial write failure propagation and state safety: `tests/test_external_compaction.py::test_external_compaction_apply_partial_write_failure_does_not_mark_state`
