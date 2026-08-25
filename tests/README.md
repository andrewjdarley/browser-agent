# Browser Use Demo - Test Suite

Comprehensive test suite for the refactored Browser Use Demo with extensive edge case coverage.

## Installation

```bash
# Install test dependencies
pip install -r test-requirements.txt

# Or install with extras
pip install -e ".[test]"
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage report
```bash
pytest tests/ --cov=browser_use_demo --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run specific test file
```bash
pytest tests/test_message_renderer.py -v
```

### Run specific test class or method
```bash
pytest tests/test_message_renderer.py::TestMessageRenderer -v
pytest tests/test_message_renderer.py::TestRenderMethod::test_render_string_message -v
```

### Run tests by marker
```bash
# Run only integration tests
pytest -m integration

# Run tests excluding integration
pytest -m "not integration"

# Run async tests
pytest -m asyncio
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and mocks
├── test_agent_sdk_bridge.py       # MCP server construction, SDK content-block parsing
├── test_batch_extract.py          # batch_extract tool: concurrency, rate-limit backoff,
│                                   # per-item error isolation, save_as behavior
├── test_browser_tool.py           # BrowserTool: browser actions, DOM diffing, screenshots
├── test_dom_script.py             # browser_dom_script.js accessibility-tree extraction
├── test_file_output.py            # file_output tool: writing files to the run directory
├── test_loop.py                   # Sampling loop: tool registration, prompt construction
├── test_message_renderer.py       # MessageRenderer: rendering all message/content types
├── test_model_config.py           # Per-role model selection and env var overrides
├── test_run_logger.py             # RunLogger: run_log.jsonl hook-driven event writing
├── test_streamlit_helpers.py      # Streamlit state/session helper functions
├── test_subagent.py               # dispatch_subagents: fan-out, error isolation
└── test_verify.py                 # verify_finding: independent claim re-derivation
```

## Test Coverage

Each file above tests its corresponding module - see the module's own docstrings for the
specifics of what each covers. Broad areas exercised across the suite:
- Edge cases: empty/malformed inputs, missing fields, None values
- Error handling and per-item isolation (a single bad item shouldn't fail an entire batch/fan-out)
- Environment variable handling (present/missing/invalid, via `monkeypatch`)
- Mocked Playwright/Streamlit/SDK boundaries so tests run without a live browser or API calls

## Edge Cases Covered

1. **Boundary Conditions**
   - Empty strings, lists, dictionaries
   - Single item collections
   - Maximum size inputs (100k+ character messages)
   - Null/None values

2. **Type Mismatches**
   - Wrong types for expected fields
   - Missing required fields
   - Extra unexpected fields
   - Invalid message structures

3. **State Inconsistencies**
   - Tools referenced but not in session_state
   - Partially initialized state
   - Concurrent modifications
   - Corrupted state

4. **Error Conditions**
   - Import errors
   - Asyncio exceptions
   - Environment variable errors
   - Lambda evaluation failures
   - Base64 decode errors

5. **Performance Edge Cases**
   - Very large message histories (1000+ messages)
   - Deeply nested content (100+ levels)
   - Circular references
   - Unicode and special characters

## Mocking Strategy

### Streamlit Components
All Streamlit components are mocked to enable testing without a running Streamlit server:
- `st.session_state`
- `st.chat_message`
- `st.markdown`, `st.write`, `st.error`, `st.code`, `st.image`
- `st.chat_input`, `st.stop`

### External Dependencies
- `BrowserTool` - Mocked to avoid Playwright dependencies
- `asyncio` event loops - Mocked for controlled testing
- Environment variables - Mocked via `monkeypatch`

## Fixtures

Key fixtures provided in `conftest.py`:

- `mock_streamlit` - Complete Streamlit mocking setup
- `mock_browser_tool` - BrowserTool mock
- `sample_tool_result` - Various ToolResult configurations
- `sample_messages` - Diverse message structures for testing
- `edge_case_messages` - Messages designed to trigger edge cases
- `mock_asyncio_loop` - Controlled event loop for testing
- `mock_environment` - Environment variable setup
- `clean_environment` - Remove environment variables

## Continuous Integration

To run tests in CI:

```bash
# Install dependencies
pip install -e ".[test]"

# Run tests with coverage
pytest tests/ --cov=browser_tools_api_demo --cov-report=xml --cov-report=term

# Generate coverage badge
coverage-badge -o coverage.svg
```

## Contributing

When adding new features or refactoring:
1. Add corresponding tests for new functionality
2. Ensure all edge cases are covered
3. Run the full test suite before committing
4. Maintain >95% code coverage
5. Update this README if test structure changes