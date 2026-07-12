"""Tests for the memoized SersicMixture.getProfile.

The memoization must be bit-identical to an uncached lookup (keyed on the
exact float), return the SAME object on repeated calls, and hand out
read-only arrays so any latent in-place mutator fails loudly instead of
corrupting every subsequent user of that sersic index.

Run in the `spherex` conda env:  pytest tests/test_sersic_cache.py -q
"""
import numpy as np
import pytest

from tractor_jax.sersic import SersicMixture


def fresh_profile(sindex):
    """Uncached reference: a brand-new table instance every call."""
    return SersicMixture()._getProfile(sindex)


# Index grid covering: below-lowest clamp, table knots, ramp overlaps between
# component-count ranges, the sindex > 1 core branch, above-highest clamp.
INDEX_GRID = [0.25, 0.29, 0.3, 0.35, 0.4, 0.41, 0.5, 0.55, 0.6, 0.7, 0.75,
              0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 6.19,
              6.3, 8.0]


@pytest.mark.parametrize("sindex", INDEX_GRID)
def test_cached_equals_fresh(sindex):
    cached = SersicMixture.getProfile(sindex)
    ref = fresh_profile(sindex)
    assert np.array_equal(np.asarray(cached.amp), np.asarray(ref.amp))
    assert np.array_equal(np.asarray(cached.mean), np.asarray(ref.mean))
    assert np.array_equal(np.asarray(cached.var), np.asarray(ref.var))


def test_second_call_returns_same_object():
    a = SersicMixture.getProfile(1.7)
    b = SersicMixture.getProfile(1.7)
    assert a is b
    # numpy scalar with the same float value hits the same entry
    c = SersicMixture.getProfile(np.float64(1.7))
    assert c is a


def test_distinct_indices_distinct_entries():
    a = SersicMixture.getProfile(1.7)
    b = SersicMixture.getProfile(1.7000001)
    assert a is not b


def test_cached_arrays_are_readonly():
    prof = SersicMixture.getProfile(2.3)
    for arr in (prof.amp, prof.mean, prof.var):
        arr = np.asarray(arr)
        assert not arr.flags.writeable
        with pytest.raises((ValueError, RuntimeError)):
            arr[(0,) * arr.ndim] = 0.0


def test_downstream_transforms_unaffected():
    """apply_shear / apply_affine on a cached profile must still work (they
    allocate fresh var arrays; amp/mean are shared read-only views)."""
    prof = SersicMixture.getProfile(4.0)
    T = np.array([[1.2, 0.1], [-0.1, 0.9]])
    sheared = prof.apply_shear(T)
    assert sheared is not prof
    assert sheared.var.flags.writeable          # fresh output array
    # rebinding-style amplitude scaling (the engine idiom) leaves the cached
    # object untouched
    scaled_amp = sheared.amp * 0.5
    assert np.array_equal(np.asarray(prof.amp) * 0.5, scaled_amp)
