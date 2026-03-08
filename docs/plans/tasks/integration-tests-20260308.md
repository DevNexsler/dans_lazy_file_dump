---
name: Implement production-readiness integration tests
type: test-implementation
---

## Objective

Implement test cases defined in skeleton files to create a production-readiness test suite. If these tests pass, the system is safe to deploy.

## Target Files

- Skeleton: tests/test_api_server.int.test.py (REST API - 12 tests)
- Skeleton: tests/test_config_edge_cases.int.test.py (config edge cases - 3 tests)
- Skeleton: tests/test_mcp_handlers.int.test.py (MCP handlers - 5 tests)
- Skeleton: tests/test_indexing_roundtrip.int.test.py (full pipeline - 4 tests)
- Skeleton: tests/test_provider_errors.int.test.py (error handling - 4 tests)
- Skeleton: tests/test_production_readiness.e2e.test.py (E2E lifecycle - 2 tests)
- Design Doc: docs/design/system-design-doc.md

## Key Implementation Constraints

- All integration tests use real LanceDB with mock embedding providers (no API keys needed)
- REST API tests use httpx.AsyncClient with ASGITransport for in-process testing
- E2E tests combine REST API + MCP handlers + LanceDB in a single journey
- Must not duplicate coverage from existing 14 test files

## Tasks

- [x] Implement tests/test_api_server.int.test.py (upload, download, list, auth, path traversal)
- [x] Implement tests/test_config_edge_cases.int.test.py (env var overrides, validation)
- [x] Implement tests/test_mcp_handlers.int.test.py (search, list, facets through real store)
- [x] Implement tests/test_indexing_roundtrip.int.test.py (extract -> chunk -> store -> search)
- [x] Implement tests/test_provider_errors.int.test.py (timeouts, bad responses, failures)
- [x] Implement tests/test_production_readiness.e2e.test.py (full lifecycle + resilience)
- [x] Verify all tests pass
- [x] Ensure no quality issues

## Acceptance Criteria

- All skeleton test cases implemented with real assertions
- All tests passing without requiring external API keys
- No quality issues (linting, type errors)
- Tests provide deployment confidence when passing
