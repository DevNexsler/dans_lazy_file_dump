---
name: Implement full E2E live tests with real providers
type: test-implementation
---

## Objective

Implement true end-to-end tests that exercise the complete system with real cloud providers (OpenRouter, DeepInfra). These tests confirm production readiness when run with API keys.

## Target Files

- Skeleton: tests/test_full_e2e_live.py
- Design Doc: docs/design/system-design-doc.md

## Tasks

- [x] Implement test_upload_index_search_download_with_real_providers
- [x] Implement test_multi_document_search_quality_with_reranker
- [x] Implement test_mcp_metadata_handlers_with_real_indexed_data
- [x] Implement test_taxonomy_guided_enrichment_with_real_providers
- [x] Verify all tests pass with live API keys
- [x] Ensure no quality issues

## Acceptance Criteria

- All 4 tests pass with real API keys (OPENROUTER_API_KEY, DEEPINFRA_API_KEY)
- All tests marked @pytest.mark.live and skip gracefully without keys
- REST API tested via httpx AsyncClient with ASGITransport
- MCP handlers called directly (not through transport)
- Enrichment fields verified (enr_summary, enr_topics populated)
- Reranker verified (diagnostics.reranker_applied == True)
- Search quality: relevant docs rank higher than irrelevant
- No mocks — every provider is real
