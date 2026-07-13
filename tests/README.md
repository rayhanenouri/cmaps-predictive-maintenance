# Tests

Unit and integration tests for the predictive maintenance pipeline.

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_data_loader

# Run with verbose output
python -m unittest discover tests -v
```

## Test Coverage

- `test_data_loader.py` - Data loading and RUL calculation
- (Future) `test_feature_engineering.py` - Feature transformations
- (Future) `test_model.py` - Model training and prediction

## Note

Some integration tests require the NASA C-MAPS dataset to be present in the `data/` directory.
