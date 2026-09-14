# SENTRA ML Training Pipeline - Test Suite

Comprehensive test suite for the SENTRA ML training pipeline.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_secret_sharing.py   # Secret sharing tests
├── test_kvs.py              # Key-Value Store tests
├── test_beaver_triples.py   # Beaver triple tests
├── test_secure_comparison.py # Secure comparison tests
├── test_secure_division.py  # Secure division tests
├── test_mpc_engine.py       # MPC engine tests
├── test_secure_matrix_ops.py # Matrix operations tests
├── test_communication.py    # Communication protocol tests
├── test_integration.py      # Integration tests
└── test_benchmarks.py       # Performance benchmarks
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest node/tests/test_secret_sharing.py
```

### Run specific test
```bash
pytest node/tests/test_secret_sharing.py::TestShamirSecretSharing::test_share_secret
```

### Run with coverage
```bash
pytest --cov=ml_training --cov-report=html
```

### Run only fast tests (exclude slow benchmarks)
```bash
pytest -m "not slow"
```

### Run only unit tests (exclude integration)
```bash
pytest -m "not integration"
```

### Run benchmarks
```bash
pytest node/tests/test_benchmarks.py -m benchmark
```

## Test Categories

### Unit Tests
- **Secret Sharing**: Shamir and Packed Shamir secret sharing
- **KVS**: Key-Value Store operations
- **Beaver Triples**: Triple generation and secure multiplication
- **Secure Operations**: Comparison, division, clipping
- **MPC Engine**: Forward/backward pass, weight updates
- **Matrix Operations**: Secure matrix multiplication
- **Communication**: Network protocol, reconstruction

### Integration Tests
- **Training Pipeline**: End-to-end training workflow
- **Coordinator**: Mini-batch selection, weight initialization

### Benchmarks
- **Secret Sharing**: Sharing and reconstruction performance
- **Beaver Triples**: Triple generation and multiplication
- **Matrix Operations**: Matrix multiplication performance
- **End-to-End**: Full training pipeline performance

## Test Coverage

The test suite covers:
- ✅ All core components
- ✅ Secure operations
- ✅ Matrix operations
- ✅ Communication protocol
- ✅ Integration scenarios

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-benchmark pytest-cov
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:
```bash
# Run tests with coverage
pytest --cov=ml_training --cov-report=xml

# Run with verbose output
pytest -v

# Run with parallel execution
pytest -n auto
```

## Writing New Tests

When adding new functionality:

1. **Add unit tests** in the appropriate test file
2. **Add integration tests** if testing multiple components
3. **Add benchmarks** for performance-critical code
4. **Update this README** with new test categories

### Example Test Structure

```python
class TestNewFeature:
    """Tests for NewFeature"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        feature = NewFeature()
        
        # Act
        result = feature.do_something()
        
        # Assert
        assert result is not None
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/sentra
pytest
```

### Fixture Errors
If fixtures are not found, check `conftest.py` is in the `tests/` directory.

### Slow Tests
Mark slow tests with `@pytest.mark.slow` and exclude with `-m "not slow"`.

