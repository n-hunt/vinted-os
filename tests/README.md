# DocuFlow Library Tests

This directory contains unit tests for the DocuFlow library modules.

## Test Coverage

- `test_vision.py` - PDFProcessor image manipulation tests
- `test_parser.py` - TransactionParser extraction tests  
- `test_generator.py` - LabelGenerator PDF creation tests

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_vision.py -v

# Run with coverage
pytest tests/ --cov=src/docuflow
```

## Test Data

Place sample PDFs in `tests/fixtures/` for integration testing.
