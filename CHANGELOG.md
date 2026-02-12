# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Policy

- **MAJOR** version bumps indicate incompatible API changes or breaking behavioral changes
- **MINOR** version bumps indicate new functionality added in a backwards-compatible manner
- **PATCH** version bumps indicate backwards-compatible bug fixes

### Pre-1.0 Policy

While in beta (0.x.y), the API is considered unstable:
- MINOR bumps may introduce breaking changes
- PATCH bumps are safe to upgrade within the same minor version

## [Unreleased]

### Added
- OpenAI Codex source support (AR-001)
- Incremental sync for evolving sessions with checkpoint persistence (AR-002)
- Production hardening for Codex ingestion with comprehensive fixture-based testing (AR-005)
- Package metadata finalized for production adoption (AR-008)
- CI pipeline for automated testing across Python 3.11, 3.12, and 3.13

### Changed
- Development status upgraded from Alpha to Beta

## [0.1.0] - 2025-02-12

### Added
- Initial release with core functionality:
  - Session management (start, log, end)
  - Knowledge compaction with LLM synthesis
  - Context retrieval for new sessions
  - Cursor source integration (SQLite + JSONL)
  - Claude Code source integration
  - OpenCode source integration
  - TUI with command palette and settings
  - Configuration management
  - Provider support (Anthropic, OpenAI, Google, Ollama, vLLM, LM Studio)

[Unreleased]: https://github.com/natecard/agent-recall/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/natecard/agent-recall/releases/tag/v0.1.0
