# Tests

Tests for the data loading and RUL computation pipeline.

## Run

```bash
python -m pytest tests/ -v
```

Requires NASA C-MAPS FD001 files in `data/` directory.
Tests are skipped automatically if data files are not present.
