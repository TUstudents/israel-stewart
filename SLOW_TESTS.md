# Slow Evolution Tests

## Overview

Some tests in the test suite involve full time evolution over many timesteps (50-100+ steps) and are marked with `@pytest.mark.slow`. These tests are important for validating long-term numerical accuracy but are computationally intensive.

## Marked Slow Tests

### 1. Eigenmode Preservation (`tests/test_eigenmode_preservation.py`)
- **Test**: `test_eigenmode_ratios_are_preserved`
- **Duration**: ~40 seconds
- **Grid**: 32³ × 16
- **Evolution**: 100 timesteps (t=0 to t=1.0, dt=0.01)
- **Validates**: Eigenmode structure preservation during sound wave propagation

### 2. Sound Wave Energy Conservation (`israel_stewart/tests/test_dynamic_conservation.py`)
- **Test**: `TestPhysicalScenarios::test_sound_wave_energy_conservation`
- **Duration**: ~60 seconds
- **Grid**: 32³ (high resolution)
- **Evolution**: ~50 timesteps (half wave period)
- **Validates**: Energy conservation during acoustic propagation

### 3. Diffusion Particle Conservation (`israel_stewart/tests/test_dynamic_conservation.py`)
- **Test**: `TestPhysicalScenarios::test_diffusion_conserves_particles`
- **Duration**: ~45 seconds
- **Grid**: 16³
- **Evolution**: 100 timesteps
- **Validates**: Particle conservation with active diffusion

## Running Tests

### Skip Slow Tests (Default for CI/CD)
```bash
# Skip all slow tests (recommended for routine testing)
uv run pytest -m "not slow"

# Or explicitly with the whole test suite
uv run pytest tests/ israel_stewart/tests/ -m "not slow"
```

### Run Only Slow Tests
```bash
# Run only the slow evolution tests
uv run pytest -m slow

# Run slow tests without coverage (faster)
uv run pytest -m slow --no-cov

# Run specific slow test
uv run pytest tests/test_eigenmode_preservation.py -m slow -v
```

### Run All Tests (Including Slow)
```bash
# Run everything (may take 5+ minutes)
uv run pytest tests/ israel_stewart/tests/

# With increased timeout for slow tests
uv run pytest tests/ israel_stewart/tests/ --timeout=300
```

## CI/CD Integration

For continuous integration, **skip slow tests by default** to keep build times reasonable:

```yaml
# .github/workflows/test.yml
- name: Run fast tests
  run: uv run pytest -m "not slow" --timeout=120

# Optional: separate job for slow tests (nightly build)
- name: Run slow evolution tests
  run: uv run pytest -m slow --timeout=600
  if: github.event_name == 'schedule'  # Only on nightly builds
```

## Adding New Slow Tests

To mark a test as slow:

```python
import pytest

@pytest.mark.slow
def test_long_evolution():
    """Test that requires many timesteps or long simulation time."""
    # Evolution code here...
```

**Criteria for marking as slow**:
- Evolution > 50 timesteps
- Grid resolution > 32³
- Test duration > 30 seconds
- Tests critical for validation but not needed for every commit

## Notes

- Slow tests currently have some failures (eigenmode drift, constraint violations) that are under investigation
- These tests validate **long-term numerical stability**, not just initial conditions
- Use slow tests for:
  - Pre-release validation
  - Performance regression testing
  - Nightly CI builds
  - Local validation of major changes

## Current Status (Phase 16)

As of Phase 16, slow evolution tests are:
- ✅ Properly marked with `@pytest.mark.slow`
- ✅ Can be selectively run/skipped
- ⚠️ Some tests failing (under investigation):
  - Eigenmode drift exceeds 15% tolerance
  - Constraint violations in some scenarios
- 🔄 Integration with full Phase 16 validation infrastructure

See `PHASE_16_SUMMARY.md` for comprehensive validation results.
