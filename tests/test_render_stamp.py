"""Compact-stamp rendering (opt-in ``render_stamp``) against the full grid.

The padded-batch builder can emit the PSF transform on a small ``S x S``
high-res stamp and flag each galaxy as compact or large; the template
renderer then draws point sources and compact galaxies on the stamp and only
the large galaxies on the full padded tile grid.  These tests pin down

* that the default bundle and the default solver path are untouched
  (``render_stamp=None`` emits no stamp keys; ``render_mode="full"`` on a
  stamp bundle reproduces the full-grid templates bit for bit);
* that the stamp path agrees with the full grid to the periodic-sinc level
  (1e-4 of the template peak here; the production kernel does ~1e-5) for
  every compact source, including ones whose stamp is clipped by the tile
  edge, and that large galaxies take the full-grid path exactly;
* the static PSF-type dispatch (``psf_type="fft"``) equals the historical
  ``jnp.where`` path exactly;
* that the jitted vmapped solver accepts the stamp bundle and recovers the
  same fluxes as the full bundle on a noiseless image.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tractor_jax.jax import batching as tjb
from tractor_jax.jax.optimizer import _render_source_templates

PIXSCALE_ARCSEC = 6.15
CD_INV = np.eye(2) * (3600.0 / PIXSCALE_ARCSEC)        # px per degree
H = W = 21
S_STAMP = 80


class _Cat(dict):
    """Minimal catalog: column access by name, row count via len()."""
    def __len__(self):
        return len(self["shape_r"])


def _gaussian_psf(n=51, sigma_hr=4.0):
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2
    k = np.exp(-0.5 * ((yy - c) ** 2 + (xx - c) ** 2) / sigma_hr ** 2)
    return k / k.sum()


def _catalog():
    # point sources (shape_r == 0) and galaxies; the last galaxy is large
    shape_r = np.array([0.0, 0.0, 0.0, 0.8, 1.5, 0.5, 30.0])
    sersic = np.array([0.0, 0.0, 0.0, 1.0, 4.0, 2.0, 4.0])
    shape_ab = np.array([1.0, 1.0, 1.0, 0.7, 0.5, 0.9, 0.8])
    shape_phi = np.array([0.0, 0.0, 0.0, 30.0, 120.0, 75.0, 10.0])
    # parent-frame positions inside BOTH views' boxes (view 0 at origin (0, 0),
    # view 1 at (12, 3): x in [12, 21), y in [3, 21)); view 1 sees them at
    # different sub-pixel phases and some stamps clip a tile edge there
    sx = np.array([12.3, 16.6, 20.4, 13.2, 18.9, 12.7, 16.0])
    sy = np.array([4.1, 9.2, 3.6, 15.5, 20.4, 18.8, 12.5])
    cat = _Cat(shape_r=shape_r, sersic=sersic, shape_ab=shape_ab,
               shape_phi=shape_phi)
    return cat, sx, sy


def _views(psf, n_views=2):
    rng = np.random.default_rng(0)
    out = []
    origins = [(0.0, 0.0), (12.0, 3.0)]
    for i in range(n_views):
        out.append({
            "data": rng.normal(size=(H, W)).astype(np.float32),
            "invvar": np.ones((H, W), np.float32),
            "psf": psf,
            "src_indices": list(range(7)),
            "origin": origins[i],
        })
    return out


def _build(render_stamp=None, **kw):
    psf = _gaussian_psf()
    cat, sx, sy = _catalog()
    return tjb.build_padded_batches(
        _views(psf), cat, sx, sy, psf_sampling=0.2, fixed_max_factor=5.0,
        fit_background=True, cd_inv=CD_INV, render_stamp=render_stamp, **kw)


def _single(bundle, i):
    imgd = jax.tree_util.tree_map(lambda a: a[i], bundle.images_data)
    bat = {}
    for key, val in bundle.batches.items():
        axes = bundle.in_axes[key]
        bat[key] = jax.tree_util.tree_map(
            lambda a, ax: a[i] if ax == 0 else a, val, axes)
    return imgd, bat


def _templates(bundle, i, sampling_factor=5.0, **kw):
    imgd, bat = _single(bundle, i)
    n_flux = bundle.initial_fluxes.shape[1]
    return np.asarray(_render_source_templates(imgd, bat, n_flux,
                                               sampling_factor=sampling_factor,
                                               **kw))


def test_stamp_engages_without_an_explicit_sampling_factor():
    """The batched solvers call the renderer with ``sampling_factor=None``
    (the factor then comes from ``psf_data['sampling']``); the stamp path
    must engage on that call signature too and agree with the full grid."""
    b_full = _build()
    b_stamp = _build(render_stamp=S_STAMP)
    tile = (slice(0, H), slice(0, W))
    for i in range(2):
        t_full = _templates(b_full, i, sampling_factor=None)
        t_stamp = _templates(b_stamp, i, sampling_factor=None)
        # the two full-grid conventions (explicit 5.0 / None) coincide here
        assert np.array_equal(t_full, _templates(b_full, i))
        for slot in range(t_full.shape[0]):
            peak = np.abs(t_full[slot]).max()
            if peak == 0.0:
                continue
            diff = np.abs(t_stamp[slot][tile] - t_full[slot][tile]).max() / peak
            assert diff < 1e-5, f"slot {slot}: {diff}"
    # and the stamp path really ran: the clipped low-edge source differs from
    # the full grid in the zero-weight padding (see the docstring above)
    assert not np.array_equal(_templates(b_stamp, 1, sampling_factor=None),
                              _templates(b_full, 1, sampling_factor=None))


def test_default_bundle_has_no_stamp_keys():
    b = _build()
    assert "fft_stamp" not in b.images_data["psf"]
    for key in tjb._OPTIONAL_GALAXY_KEYS:
        assert key not in b.batches["Galaxy"]
    assert b.meta["render_stamp"] is None
    assert b.in_axes["Galaxy"] == tjb._IN_AXES_TEMPLATES["Galaxy"]


def test_stamp_bundle_structure_and_split():
    b = _build(render_stamp=S_STAMP)
    n_views = len(_views(_gaussian_psf()))
    assert b.images_data["psf"]["fft_stamp"].shape == (n_views, S_STAMP, S_STAMP // 2 + 1)
    gal = b.batches["Galaxy"]
    for key in tjb._OPTIONAL_GALAXY_KEYS:
        assert key in gal and b.in_axes["Galaxy"][key] == 0
    assert b.meta["render_stamp"] == S_STAMP
    # the 30" n=4 galaxy is the only large one; it is slot 3 of the galaxy
    # block in both views (catalog order 3,4,5,6 -> slots 0..3)
    stamp_mask = np.asarray(gal["stamp_mask"])
    large_idx = np.asarray(gal["large_idx"])
    large_mask = np.asarray(gal["large_mask"])
    for v in range(n_views):
        assert stamp_mask[v, :3].tolist() == [1.0, 1.0, 1.0]
        assert stamp_mask[v, 3] == 0.0
        assert large_mask[v, 0] == 1.0 and large_idx[v, 0] == 3
        assert large_mask[v, 1:].sum() == 0.0
    assert b.meta["n_large_max"] == 8          # one bucket


def test_render_mode_full_on_stamp_bundle_is_bit_identical():
    b_full = _build()
    b_stamp = _build(render_stamp=S_STAMP)
    for i in range(2):
        t_full = _templates(b_full, i)
        t_forced = _templates(b_stamp, i, render_mode="full")
        assert np.array_equal(t_full, t_forced)


def test_stamp_matches_full_grid_for_every_source():
    """Inside the tile the two paths agree to the periodic-sinc level; the
    only difference lives in the zero-weight padding region, where the full
    grid's periodic wrap puts light from sources near the low edge that the
    stamp path (physically) clips instead."""
    b_full = _build()
    b_stamp = _build(render_stamp=S_STAMP)
    gal_large_slot = b_full.meta["max_ps"] + 3
    for i in range(2):
        t_full = _templates(b_full, i)
        t_stamp = _templates(b_stamp, i)           # render_mode="auto" -> stamp
        assert t_full.shape == t_stamp.shape
        tile = (slice(0, H), slice(0, W))          # the weighted region
        for slot in range(t_full.shape[0]):
            peak = np.abs(t_full[slot]).max()
            if peak == 0.0:                          # padding slot
                assert np.abs(t_stamp[slot]).max() == 0.0
                continue
            diff = np.abs(t_stamp[slot][tile] - t_full[slot][tile]).max() / peak
            if slot == gal_large_slot:
                assert diff < 1e-6, f"large galaxy slot {slot}: {diff}"
                assert np.array_equal(t_stamp[slot], t_full[slot])
            else:
                assert diff < 1e-5, f"slot {slot}: {diff}"
            s_full, s_stamp = t_full[slot][tile].sum(), t_stamp[slot][tile].sum()
            assert abs(s_stamp - s_full) <= 1e-5 * max(abs(s_full), peak)
        # the background template is untouched by the option
        bg = b_full.meta["bg_idx"]
        assert np.array_equal(t_stamp[bg], t_full[bg])


def test_render_mode_stamp_requires_the_stamp_transform():
    b_full = _build()
    imgd, bat = _single(b_full, 0)
    with pytest.raises(ValueError):
        _render_source_templates(imgd, bat, b_full.initial_fluxes.shape[1],
                                 sampling_factor=5.0, render_mode="stamp")
    with pytest.raises(ValueError):
        _render_source_templates(imgd, bat, b_full.initial_fluxes.shape[1],
                                 sampling_factor=5.0, render_mode="bogus")


def test_static_fft_dispatch_equals_where_path():
    b = _build()
    for i in range(2):
        t_where = _templates(b, i)
        t_fft = _templates(b, i, psf_type="fft")
        assert np.array_equal(t_where, t_fft)


def test_stamp_too_small_or_misaligned_raises():
    with pytest.raises(ValueError):
        _build(render_stamp=40)            # kernel is 51 high-res px
    with pytest.raises(ValueError):
        _build(render_stamp=82)            # not a multiple of the sampling 5


def test_batched_solver_recovers_fluxes_on_stamp_bundle():
    b_full = _build()
    b_stamp = _build(render_stamp=S_STAMP)
    n_flux = b_full.initial_fluxes.shape[1]
    truth = np.zeros(n_flux, np.float32)
    truth[:b_full.meta["max_ps"]] = [3.0, 5.0, 2.0]
    truth[b_full.meta["max_ps"]:b_full.meta["max_ps"] + 4] = [4.0, 6.0, 1.5, 8.0]
    truth[b_full.meta["bg_idx"]] = 0.2
    # noiseless model images from the FULL-grid templates
    imgs = []
    for i in range(2):
        t = _templates(b_full, i)
        imgs.append(np.tensordot(truth, t, axes=(0, 0)))
    # templates are rendered on the padded data grid already, so the model
    # images are padded-size too
    data_pad = np.stack(imgs).astype(np.float32)
    assert data_pad.shape == tuple(b_full.images_data["data"].shape)

    def solve(bundle):
        imgd = dict(bundle.images_data)
        imgd["data"] = jnp.asarray(data_pad)
        fn = tjb.make_batched_solver("linear", in_axes=bundle.in_axes,
                                     rcond=1e-12, cache=False)
        f, v = fn(bundle.initial_fluxes, imgd, bundle.batches)
        return np.asarray(f)

    f_full = solve(b_full)
    f_stamp = solve(b_stamp)
    live = truth != 0
    for i in range(2):
        assert np.allclose(f_full[i][live], truth[live], rtol=2e-3, atol=2e-3)
        assert np.allclose(f_stamp[i][live], truth[live], rtol=2e-3, atol=2e-3)
        assert np.allclose(f_stamp[i][live], f_full[i][live], rtol=2e-3, atol=2e-3)
