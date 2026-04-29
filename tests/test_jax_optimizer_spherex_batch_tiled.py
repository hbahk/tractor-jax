"""
Forced photometry for the new spherex-retrieval cutout format using tiled
batched JAX optimisation.

The new SPHEREx data delivered by ``spherex_retrieval.retrieve`` is one
multi-extension FITS file per overlapping pointing (see
.claude/spherex_data_desc/spherex_cutouts.md and
repos/spherex-retrieval/README.md). Each MEF carries:

    PRIMARY  IMAGE  FLAGS  VARIANCE  ZODI  PSF  PSF_ZONES  [CWAVE]  [CBAND]  [SAPM]

The cutouts in this field are 100x100 native pixels, so we split each one into
~15-pixel core tiles with a halo and batch every tile from every cutout into a
single ``vmap(solve_fluxes_linear)`` call. Per-cutout photometry of the main
source is read from the tile that contains the source.

This mirrors tests/test_jax_optimizer_spherex_batch.py in spirit and shares the
same scaling / masking / background-fitting logic, but consumes the new
per-cutout MEF layout and adds tile-level batching.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import time
from functools import partial
from pathlib import Path

import astropy.units as u
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from photutils.background import Background2D
from tqdm import tqdm

from tractor_jax.jax.optimizer import solve_fluxes_linear
from tractor_jax.sersic import SersicMixture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPHEREX_PIXSCALE = 6.15  # arcsec / native pixel
PIXAREA_CONST_SR = ((SPHEREX_PIXSCALE * u.arcsec) ** 2).to_value(u.sr)
ARCSEC2_TO_SR = (1.0 * u.arcsec ** 2).to_value(u.sr)
IMG_SCALE = 1.0e9  # MJy -> mJy for numerical stability

# Tiling defaults — user asked for ~15 native pixels with a halo equal to the
# native PSF half-width.
TILE_SIZE = 15
TILE_HALO = 5

# Background fitting (run once per cutout, not per tile).
BKG_MODEL = "photutils"  # "photutils", "plane", or "none"
BKG_BOX_SIZE = 10
BKG_FILTER_SIZE = 3

# Bit definitions from the FLAGS header (matches the legacy script).
FLAG_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "PHANTOM": 4,
    "REFERENCE": 5, "NONFUNC": 6, "DICHROIC": 7, "MISSING_DATA": 9,
    "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "SOURCE": 21,
}
MASK_FLAGS = ["SUR_ERROR", "PHANMISS", "NONFUNC", "MISSING_DATA",
              "HOT", "COLD", "PERSIST", "OUTLIER"]
MASKBITS = 0
for _name in MASK_FLAGS:
    MASKBITS |= (1 << FLAG_BITS[_name])
SOURCE_BIT = 1 << FLAG_BITS["SOURCE"]


# ---------------------------------------------------------------------------
# PSF zone lookup (mirrors tests/utils.py / repos/spherex-retrieval/psf.py).
# In the new format the PSF cube is already restricted to overlapping zones, so
# the lookup table is small (often a single row for 100x100 cutouts).
# ---------------------------------------------------------------------------

def cutout_to_orig(x_cut, y_cut, *, crpix1a, crpix2a):
    """Map a 0-based cutout pixel coord to a 0-based detector pixel coord.

    CRPIX*A in the new bundle headers are 1-based pixel positions of the
    cutout's (0,0) pixel in the original detector frame, matching the
    convention used by repos/spherex-retrieval.
    """
    return (1.0 + (x_cut - crpix1a), 1.0 + (y_cut - crpix2a))


def select_zone_plane(psf_zones_tab, x_orig, y_orig):
    """Return the local plane index (0-based into psf_cube) closest to (x_orig, y_orig)."""
    dx = psf_zones_tab["x"] - x_orig
    dy = psf_zones_tab["y"] - y_orig
    return int(psf_zones_tab["plane_idx"][np.argmin(dx * dx + dy * dy)])


def downsample_psf_oversample2(psf):
    """Downsample 2x while preserving center and total sum (10x -> 5x oversample)."""
    h, w = psf.shape
    cy, cx = h // 2, w // 2
    oh, ow = h // 2 + 1, w // 2 + 1
    ocy, ocx = oh // 2, ow // 2

    out = np.zeros((oh, ow), dtype=psf.dtype)
    out[0:ocy, 0:ocx] = 0.25 * (
        psf[0:cy:2, 0:cx:2] + psf[1:cy:2, 0:cx:2]
        + psf[0:cy:2, 1:cx:2] + psf[1:cy:2, 1:cx:2]
    )
    out[0:ocy, ocx + 1:ow] = 0.25 * (
        psf[0:cy:2, cx + 1:w:2] + psf[1:cy:2, cx + 1:w:2]
        + psf[0:cy:2, cx + 2:w:2] + psf[1:cy:2, cx + 2:w:2]
    )
    out[ocy + 1:oh, 0:ocx] = 0.25 * (
        psf[cy + 1:h:2, 0:cx:2] + psf[cy + 2:h:2, 0:cx:2]
        + psf[cy + 1:h:2, 1:cx:2] + psf[cy + 2:h:2, 1:cx:2]
    )
    out[ocy + 1:oh, ocx + 1:ow] = 0.25 * (
        psf[cy + 1:h:2, cx + 1:w:2] + psf[cy + 2:h:2, cx + 1:w:2]
        + psf[cy + 1:h:2, cx + 2:w:2] + psf[cy + 2:h:2, cx + 2:w:2]
    )
    out[ocy, 0:ocx] = 0.5 * (psf[cy, 0:cx:2] + psf[cy, 1:cx:2])
    out[ocy, ocx + 1:ow] = 0.5 * (psf[cy, cx + 1:w:2] + psf[cy, cx + 2:w:2])
    out[0:ocy, ocx] = 0.5 * (psf[0:cy:2, cx] + psf[1:cy:2, cx])
    out[ocy + 1:oh, ocx] = 0.5 * (psf[cy + 1:h:2, cx] + psf[cy + 2:h:2, cx])
    out[ocy, ocx] = psf[cy, cx]

    total = psf.sum()
    out_sum = out.sum()
    if out_sum != 0:
        out *= total / out_sum
    return out


# ---------------------------------------------------------------------------
# Background fitting (works on the full 100x100 cutout, before tiling).
# ---------------------------------------------------------------------------

def build_background_mask(flg, var, maskbits=MASKBITS, source_bit=SOURCE_BIT):
    bad = (flg & maskbits) != 0
    source = (flg & source_bit) != 0
    valid_var = np.isfinite(var) & (var > 0)
    return bad | source | (~valid_var)


def fit_background_plane(img, bkg, flg, var):
    if img.size == 0:
        return bkg
    mask = build_background_mask(flg, var)
    valid = ~mask
    if not np.any(valid):
        return bkg
    z = (img - bkg)[valid].ravel()
    w = (1.0 / var[valid]).ravel()
    yy, xx = np.indices(img.shape)
    x = xx[valid].ravel().astype(np.float64)
    y = yy[valid].ravel().astype(np.float64)
    A = np.stack([np.ones_like(x), x, y], axis=1)
    Aw = A * w[:, None]
    AtAw = A.T @ Aw
    AtAz = A.T @ (z * w)
    try:
        a, b, c = np.linalg.solve(AtAw, AtAz)
    except np.linalg.LinAlgError:
        return bkg
    plane = (a + b * xx + c * yy).astype(bkg.dtype, copy=False)
    return bkg + plane


def fit_background_photutils(img, bkg, flg, var,
                             box_size=BKG_BOX_SIZE, filter_size=BKG_FILTER_SIZE):
    if img.size == 0 or img.shape[0] < box_size or img.shape[1] < box_size:
        return bkg
    mask = build_background_mask(flg, var)
    residual = img - bkg
    try:
        bkg2d = Background2D(
            residual,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            mask=mask,
        )
    except Exception:
        return bkg
    return bkg + bkg2d.background.astype(bkg.dtype, copy=False)


def _fit_background_for_frame(args):
    img, bkg, flg, var, bkg_model = args
    if bkg_model == "photutils":
        return fit_background_photutils(img, bkg, flg, var)
    if bkg_model == "plane":
        return fit_background_plane(img, bkg, flg, var)
    return bkg


# ---------------------------------------------------------------------------
# Tiling helpers — split a cutout into tile_size cores with a halo, padding any
# out-of-bounds region with zeros (matches tractor_jax.jax.tiling.tile_image
# semantics but operates directly on numpy arrays).
# ---------------------------------------------------------------------------

def iter_tiles(H, W, tile_size, halo):
    """Yield tile metadata covering an H x W cutout.

    Every tile has the same shape ``(tile_size + 2*halo, tile_size + 2*halo)``
    so they can be stacked for ``vmap``; pixels that fall outside the cutout
    are zero-padded by :func:`extract_tile_region` and contribute zero invvar.
    The core box, however, is clipped to the cutout — only sources whose pixel
    position falls inside the actual data are assigned to a tile.
    """
    nx = max(1, math.ceil(W / tile_size))
    ny = max(1, math.ceil(H / tile_size))
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_size
            y0 = iy * tile_size
            core_x1 = min(x0 + tile_size, W)
            core_y1 = min(y0 + tile_size, H)
            yield {
                "ix": ix, "iy": iy,
                "core_x0": x0, "core_y0": y0,
                "core_x1": core_x1, "core_y1": core_y1,
                "x_start": x0 - halo,
                "y_start": y0 - halo,
                "x_end": x0 + tile_size + halo,
                "y_end": y0 + tile_size + halo,
            }


def extract_tile_region(arr, x_start, y_start, x_end, y_end, fill=0.0):
    """Slice arr[y_start:y_end, x_start:x_end], zero-padding out-of-bounds parts."""
    H, W = arr.shape
    th = y_end - y_start
    tw = x_end - x_start
    out = np.full((th, tw), fill, dtype=arr.dtype)
    im_x0 = max(0, x_start)
    im_y0 = max(0, y_start)
    im_x1 = min(W, x_end)
    im_y1 = min(H, y_end)
    if im_x1 > im_x0 and im_y1 > im_y0:
        out[im_y0 - y_start: im_y1 - y_start,
            im_x0 - x_start: im_x1 - x_start] = arr[im_y0:im_y1, im_x0:im_x1]
    return out


def shift_wcs(wcs, x_start, y_start):
    """Return a WCS whose pixel (0,0) maps to the original ``(x_start, y_start)``.

    Uses :meth:`astropy.wcs.WCS.slice`, which correctly shifts both ``wcs.wcs.crpix``
    and ``wcs.sip.crpix``. Hand-modifying ``wcs.wcs.crpix`` alone leaves the SIP
    polynomial referenced to the un-shifted pixel and mis-projects the new origin.
    The slice's stop is chosen large enough to cover any out-of-bounds halo pixels
    we will pad with zeros — only the CRPIX shift matters here.
    """
    return wcs.slice((slice(int(y_start), int(y_start) + 10**6),
                      slice(int(x_start), int(x_start) + 10**6)))


# ---------------------------------------------------------------------------
# Cutout I/O
# ---------------------------------------------------------------------------

CUTOUT_RE = re.compile(r"cutout_(\d{4})_(.+)_D(\d)\.fits$")


def discover_cutouts(cutouts_dir: Path):
    """Return a list of (cutout_index, path) pairs sorted by cutout index."""
    cutouts_dir = Path(cutouts_dir)
    pairs = []
    for p in cutouts_dir.glob("cutout_*.fits"):
        m = CUTOUT_RE.search(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort()
    return pairs


def read_cutout(path: Path):
    """Open one cutout MEF and return the arrays we need for forced photometry."""
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header.copy()
        img = np.array(hdul["IMAGE"].data, dtype=np.float64, copy=True)
        flg = np.array(hdul["FLAGS"].data, copy=True)
        var = np.array(hdul["VARIANCE"].data, dtype=np.float64, copy=True)
        zodi = np.array(hdul["ZODI"].data, dtype=np.float64, copy=True)
        psf_cube = np.array(hdul["PSF"].data, dtype=np.float64, copy=True)
        psf_zones = Table(hdul["PSF_ZONES"].data)
        img_hdr = hdul["IMAGE"].header.copy()

        cwave_center = None
        if "CWAVE" in hdul:
            cwave = np.asarray(hdul["CWAVE"].data, dtype=np.float64)
            cy, cx = cwave.shape[0] // 2, cwave.shape[1] // 2
            cwave_center = float(cwave[cy, cx])

        sapm = None
        if "SAPM" in hdul:
            sapm = np.asarray(hdul["SAPM"].data, dtype=np.float64)

    wcs = WCS(img_hdr).celestial
    crpix1a = float(img_hdr.get("CRPIX1A", 1))
    crpix2a = float(img_hdr.get("CRPIX2A", 1))
    psf_oversamp = int(primary.get("OVERSAMP", 10))
    detector = int(primary.get("DETECTOR", img_hdr.get("DETECTOR", -1)))
    return {
        "image": img, "flags": flg, "variance": var, "zodi": zodi,
        "psf_cube": psf_cube, "psf_zones": psf_zones,
        "wcs": wcs, "image_header": img_hdr, "primary_header": primary,
        "crpix1a": crpix1a, "crpix2a": crpix2a,
        "psf_oversamp": psf_oversamp,
        "detector": detector,
        "cwave_center": cwave_center,
        "sapm": sapm,
    }


def cutout_pixel_area_sr(cutout):
    """Return per-pixel solid angle (sr), preferring the SAPM HDU when present.

    SAPM is the standalone Solid Angle Pixel Map calibration product cropped
    to the cutout (units arcsec^2), which already absorbs SIP distortion.
    """
    if cutout.get("sapm") is not None:
        return cutout["sapm"].astype(np.float64, copy=False) * ARCSEC2_TO_SR
    # Fallback: use the spatial WCS' projected pixel area at the tangent point.
    return np.full(cutout["image"].shape,
                   cutout["wcs"].proj_plane_pixel_area().to_value(u.sr),
                   dtype=np.float64)


# ---------------------------------------------------------------------------
# Build batched JAX arrays for a list of tile records.
#
# A "tile record" looks like:
#   {
#     'data': (H, W),                  background-subtracted, scaled to mJy/pixel
#     'invvar': (H, W),                inverse variance in (mJy/pixel)^-2 units
#     'psf': (Hp, Wp),                 PSF (5x oversampled in our case)
#     'wcs': astropy WCS shifted so pixel (0,0) is the tile's lower-left,
#     'src_indices': list[int],        indices into the master catalog
#   }
#
# The output (images_data, batches, initial_fluxes) is the same shape as
# extract_model_data_direct produces, ready for vmap(solve_fluxes_linear).
# Each tile owns an independent flux vector — no flux is shared between tiles.
# ---------------------------------------------------------------------------

def extract_tiled_batches(
    tile_records,
    catalog_full,
    psf_sampling=0.2,
    fixed_max_factor=5.0,
    fit_background=True,
    profile_lookup_fn=None,
):
    n_tiles = len(tile_records)
    if n_tiles == 0:
        raise ValueError("extract_tiled_batches: tile_records is empty")

    # Determine the padded native shape (uniform across tiles by construction)
    # plus the FFT/oversampled rendering target shape.
    base_h, base_w = tile_records[0]["data"].shape
    for t in tile_records:
        if t["data"].shape != (base_h, base_w):
            raise ValueError("All tiles must share the same data shape")

    max_psf_h = max(t["psf"].shape[0] for t in tile_records)
    max_psf_w = max(t["psf"].shape[1] for t in tile_records)
    max_factor = float(fixed_max_factor)
    target_sampling = max_factor if max_factor > 1.0 else 1.0

    fft_pad_h_lr = int(math.ceil(max_psf_h / max_factor))
    fft_pad_w_lr = int(math.ceil(max_psf_w / max_factor))
    padded_h = base_h + fft_pad_h_lr
    padded_w = base_w + fft_pad_w_lr
    target_h = int(round(padded_h * max_factor))
    target_w = int(round(padded_w * max_factor))

    # Classify sources per tile and find per-type maxima.
    classification = []
    max_ps = 0
    max_gal = 0
    max_gal_mog_K = 1
    for t in tile_records:
        ps_idx, gal_idx = [], []
        for ci in t["src_indices"]:
            row = catalog_full[ci]
            if row["shape_r"] == 0:
                ps_idx.append(ci)
            else:
                gal_idx.append(ci)
                if profile_lookup_fn is not None:
                    prof = profile_lookup_fn(float(row["sersic"]))
                    max_gal_mog_K = max(max_gal_mog_K, len(prof.amp))
        classification.append((ps_idx, gal_idx))
        max_ps = max(max_ps, len(ps_idx))
        max_gal = max(max_gal, len(gal_idx))

    # Per-tile flux array layout: [point sources..., galaxies..., (background)]
    n_flux = max_ps + max_gal + (1 if fit_background else 0)
    bg_idx = max_ps + max_gal

    data_list = []
    invvar_list = []
    psf_fft_list = []
    init_flux_list = []
    src_slot_per_tile = []  # dict: catalog_index -> slot in this tile's flux vector

    ps_pos = np.zeros((n_tiles, max_ps, 2), dtype=np.float32)
    ps_fidx = np.zeros((n_tiles, max_ps), dtype=np.int32)
    ps_mask = np.zeros((n_tiles, max_ps), dtype=np.float32)

    gal_pos = np.zeros((n_tiles, max_gal, 2), dtype=np.float32)
    gal_fidx = np.zeros((n_tiles, max_gal), dtype=np.int32)
    gal_mask = np.zeros((n_tiles, max_gal), dtype=np.float32)
    gal_cd = np.tile(np.eye(2, dtype=np.float32), (n_tiles, max_gal, 1, 1))
    gal_shape = np.zeros((n_tiles, max_gal, 3), dtype=np.float32)
    gal_amp = np.zeros((n_tiles, max_gal, max_gal_mog_K), dtype=np.float32)
    gal_mean = np.zeros((n_tiles, max_gal, max_gal_mog_K, 2), dtype=np.float32)
    gal_var = np.tile(np.eye(2, dtype=np.float32),
                      (n_tiles, max_gal, max_gal_mog_K, 1, 1))

    for ti, tile in enumerate(tile_records):
        d = tile["data"]
        iv = tile["invvar"]
        h, w = d.shape

        d_pad = np.zeros((padded_h, padded_w), dtype=np.float32)
        iv_pad = np.zeros((padded_h, padded_w), dtype=np.float32)
        d_pad[:h, :w] = d
        iv_pad[:h, :w] = iv
        data_list.append(jnp.asarray(d_pad))
        invvar_list.append(jnp.asarray(iv_pad))

        psf = np.asarray(tile["psf"], dtype=np.float64)
        ph, pw = psf.shape
        local_factor = 1.0 / psf_sampling if psf_sampling < 1.0 else 1.0
        if abs(local_factor - target_sampling) > 1e-3:
            ratio = target_sampling / local_factor
            new_shape = (int(round(ph * ratio)), int(round(pw * ratio)))
            psf_j = jax.image.resize(jnp.asarray(psf), new_shape, method="lanczos3")
            s_in = float(psf.sum())
            s_out = float(jnp.sum(psf_j))
            if s_out > 0 and s_in > 0:
                psf_j = psf_j * (s_in / s_out)
            psf = np.asarray(psf_j)
            ph, pw = psf.shape
        pad_psf = np.zeros((target_h, target_w), dtype=np.float64)
        cy, cx = target_h // 2, target_w // 2
        y0_psf = cy - ph // 2
        x0_psf = cx - pw // 2
        pad_psf[y0_psf:y0_psf + ph, x0_psf:x0_psf + pw] = psf
        pad_psf = np.fft.ifftshift(pad_psf)
        psf_fft_list.append(jfft.rfft2(jnp.asarray(pad_psf)))

        ps_indices, gal_indices = classification[ti]

        slots = {}
        for k, ci in enumerate(ps_indices):
            slots[ci] = k
        for k, ci in enumerate(gal_indices):
            slots[ci] = max_ps + k
        src_slot_per_tile.append(slots)

        init_f = np.zeros(n_flux, dtype=np.float32)

        # Project source positions for this tile (sky -> shifted-WCS pixels).
        all_ci = ps_indices + gal_indices
        if all_ci:
            sub = catalog_full[all_ci] if hasattr(catalog_full, "__getitem__") else None
            ras = np.asarray([catalog_full["ra"][c] for c in all_ci], dtype=np.float64)
            decs = np.asarray([catalog_full["dec"][c] for c in all_ci], dtype=np.float64)
            sco = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
            pxs, pys = tile["wcs"].world_to_pixel(sco)
            pxs = np.atleast_1d(np.asarray(pxs, dtype=np.float64))
            pys = np.atleast_1d(np.asarray(pys, dtype=np.float64))
        else:
            pxs = np.array([], dtype=np.float64)
            pys = np.array([], dtype=np.float64)

        for k, ci in enumerate(all_ci):
            ix, iy = int(round(float(pxs[k]))), int(round(float(pys[k])))
            if 0 <= iy < h and 0 <= ix < w:
                v = float(d[iy, ix])
                init_f[slots[ci]] = v if np.isfinite(v) else 0.0

        # Point sources
        for k, ci in enumerate(ps_indices):
            ps_pos[ti, k, 0] = float(pxs[k])
            ps_pos[ti, k, 1] = float(pys[k])
            ps_fidx[ti, k] = slots[ci]
            ps_mask[ti, k] = 1.0

        # Galaxies
        wcs_t = tile["wcs"]
        try:
            cd_matrix = (np.asarray(wcs_t.wcs.cd) if hasattr(wcs_t.wcs, "cd")
                         else np.asarray(wcs_t.pixel_scale_matrix))
        except Exception:
            cd_matrix = np.eye(2) * (SPHEREX_PIXSCALE / 3600.0)
        try:
            cd_inv = np.linalg.inv(cd_matrix)
        except np.linalg.LinAlgError:
            cd_inv = np.eye(2)

        for k, ci in enumerate(gal_indices):
            row = catalog_full[ci]
            offset = len(ps_indices) + k
            gal_pos[ti, k, 0] = float(pxs[offset])
            gal_pos[ti, k, 1] = float(pys[offset])
            gal_fidx[ti, k] = slots[ci]
            gal_mask[ti, k] = 1.0
            gal_cd[ti, k] = cd_inv.astype(np.float32, copy=False)
            gal_shape[ti, k, 0] = float(row["shape_r"])
            gal_shape[ti, k, 1] = float(row["shape_ab"])
            gal_shape[ti, k, 2] = float(row["shape_phi"])
            if profile_lookup_fn is not None:
                prof = profile_lookup_fn(float(row["sersic"]))
                K = len(prof.amp)
                gal_amp[ti, k, :K] = np.asarray(prof.amp, dtype=np.float32)
                gal_mean[ti, k, :K] = np.asarray(prof.mean, dtype=np.float32)
                gal_var[ti, k, :K] = np.asarray(prof.var, dtype=np.float32)

        init_flux_list.append(init_f)

    images_data = {
        "data": jnp.stack(data_list),
        "invvar": jnp.stack(invvar_list),
        "psf": {
            "type_code": jnp.zeros(n_tiles, dtype=jnp.int32),
            "sampling": jnp.full(n_tiles, target_sampling, dtype=jnp.float32),
            "fft": jnp.stack(psf_fft_list),
            "amp": jnp.zeros((n_tiles, 1)),
            "mean": jnp.zeros((n_tiles, 1, 2)),
            "var": jnp.tile(jnp.eye(2), (n_tiles, 1, 1, 1)),
        },
    }

    batches = {}
    if max_ps > 0:
        batches["PointSource"] = {
            "flux_idx": jnp.asarray(ps_fidx),
            "pos_pix": jnp.asarray(ps_pos),
            "mask": jnp.asarray(ps_mask),
        }
    if max_gal > 0:
        batches["Galaxy"] = {
            "flux_idx": jnp.asarray(gal_fidx),
            "pos_pix": jnp.asarray(gal_pos),
            "wcs_cd_inv": jnp.asarray(gal_cd),
            "shapes": jnp.asarray(gal_shape),
            "mask": jnp.asarray(gal_mask),
            "profile": {
                "amp": jnp.asarray(gal_amp),
                "mean": jnp.asarray(gal_mean),
                "var": jnp.asarray(gal_var),
            },
        }

    if fit_background:
        # Constant per tile — same scalar slot in every flux vector.
        # `flux_idx` is shared across tiles, so map it with in_axes=None below.
        if max_ps > 0 or max_gal > 0:
            init = np.stack(init_flux_list)
        else:
            init = np.zeros((n_tiles, n_flux), dtype=np.float32)
        batches["Background"] = {"flux_idx": jnp.asarray([bg_idx], dtype=jnp.int32)}
        initial_fluxes = jnp.asarray(init, dtype=jnp.float32)
    else:
        initial_fluxes = jnp.asarray(np.stack(init_flux_list), dtype=jnp.float32)

    return images_data, batches, initial_fluxes, src_slot_per_tile


# ---------------------------------------------------------------------------
# Per-cutout solve
# ---------------------------------------------------------------------------

def build_cutout_tiles(cutout, catalog_table, *, sx_all, sy_all,
                       tile_size=TILE_SIZE, halo=TILE_HALO,
                       data_scaled=None, invvar_scaled=None, psf_native=None):
    """Construct tile records for one cutout.

    Sources are filtered by their pixel position in the cutout to those that
    fall inside the (core + halo) box of each tile.
    """
    H, W = data_scaled.shape

    # Sources that lie inside this cutout (with halo margin so PSF wings count).
    inside = ((sx_all > -halo) & (sx_all < W + halo)
              & (sy_all > -halo) & (sy_all < H + halo)
              & np.isfinite(sx_all) & np.isfinite(sy_all))
    cutout_src_indices = np.where(inside)[0]
    cutout_sx = sx_all[cutout_src_indices]
    cutout_sy = sy_all[cutout_src_indices]

    tile_records = []
    main_tile_local = None
    for ti, meta in enumerate(iter_tiles(H, W, tile_size, halo)):
        xs, ys = meta["x_start"], meta["y_start"]
        xe, ye = meta["x_end"], meta["y_end"]

        in_box = ((cutout_sx >= xs) & (cutout_sx < xe)
                  & (cutout_sy >= ys) & (cutout_sy < ye))
        idxs = cutout_src_indices[in_box].tolist()

        tile_records.append({
            "data": extract_tile_region(data_scaled, xs, ys, xe, ye, fill=0.0),
            "invvar": extract_tile_region(invvar_scaled, xs, ys, xe, ye, fill=0.0),
            "psf": psf_native,
            "wcs": shift_wcs(cutout["wcs"], xs, ys),
            "src_indices": idxs,
            "tile_meta": meta,
        })
    return tile_records


def find_main_tile(tile_records, sx_main, sy_main):
    """Return the index of the tile whose CORE box contains the main source."""
    for ti, t in enumerate(tile_records):
        meta = t["tile_meta"]
        if (meta["core_x0"] <= sx_main < meta["core_x1"]
                and meta["core_y0"] <= sy_main < meta["core_y1"]):
            return ti
    # Fallback: return the tile closest to the main source.
    cx = np.array([0.5 * (t["tile_meta"]["core_x0"] + t["tile_meta"]["core_x1"])
                   for t in tile_records])
    cy = np.array([0.5 * (t["tile_meta"]["core_y0"] + t["tile_meta"]["core_y1"])
                   for t in tile_records])
    return int(np.argmin((cx - sx_main) ** 2 + (cy - sy_main) ** 2))


def process_one_cutout(cutout, catalog_table, sco_all, main_idx, main_co,
                       *, profile_lookup_fn, psf_sampling=0.2,
                       fixed_max_factor=5.0, tile_size=TILE_SIZE,
                       halo=TILE_HALO, solve_fn=None):
    """Solve forced photometry on a single cutout's tiles. Returns (flux, ferr,
    cwave_center) for the main source."""

    img = cutout["image"]
    flg = cutout["flags"]
    var = cutout["variance"]
    bkg = cutout["zodi"]

    # Background fitting (optional refinement on top of ZODI).
    if BKG_MODEL == "photutils":
        bkg = fit_background_photutils(img, bkg, flg, var)
    elif BKG_MODEL == "plane":
        bkg = fit_background_plane(img, bkg, flg, var)

    omega_sr = cutout_pixel_area_sr(cutout).astype(img.dtype, copy=False)
    img_scaled = img * omega_sr * IMG_SCALE
    bkg_scaled = bkg * omega_sr * IMG_SCALE
    var_scaled = var * (omega_sr ** 2) * (IMG_SCALE ** 2)

    invvar = 1.0 / var_scaled
    invvar[~np.isfinite(invvar)] = 0
    invvar[(flg & MASKBITS) != 0] = 0

    data = img_scaled - bkg_scaled
    data[~np.isfinite(data)] = 0.0

    # Source pixel positions in this cutout.
    pxs, pys = cutout["wcs"].world_to_pixel(sco_all)
    sx_all = np.asarray(pxs, dtype=np.float64)
    sy_all = np.asarray(pys, dtype=np.float64)

    # Pick PSF zone for the main source (in original detector pixels).
    sx_main, sy_main = float(sx_all[main_idx]), float(sy_all[main_idx])
    x_orig, y_orig = cutout_to_orig(sx_main, sy_main,
                                    crpix1a=cutout["crpix1a"],
                                    crpix2a=cutout["crpix2a"])
    plane = select_zone_plane(cutout["psf_zones"], x_orig, y_orig)
    psf_native = downsample_psf_oversample2(cutout["psf_cube"][plane])

    tile_records = build_cutout_tiles(
        cutout, catalog_table,
        sx_all=sx_all, sy_all=sy_all,
        tile_size=tile_size, halo=halo,
        data_scaled=data, invvar_scaled=invvar, psf_native=psf_native,
    )

    images_data, batches, initial_fluxes, src_slot = extract_tiled_batches(
        tile_records, catalog_table,
        psf_sampling=psf_sampling,
        fixed_max_factor=fixed_max_factor,
        fit_background=True,
        profile_lookup_fn=profile_lookup_fn,
    )

    batches_in_axes = {}
    if "PointSource" in batches:
        batches_in_axes["PointSource"] = {"flux_idx": 0, "pos_pix": 0, "mask": 0}
    if "Galaxy" in batches:
        batches_in_axes["Galaxy"] = {
            "flux_idx": 0, "pos_pix": 0, "wcs_cd_inv": 0,
            "mask": 0, "shapes": 0,
            "profile": {"amp": 0, "mean": 0, "var": 0},
        }
    if "Background" in batches:
        batches_in_axes["Background"] = {"flux_idx": None}

    if solve_fn is None:
        solve_fn = jax.jit(jax.vmap(
            partial(solve_fluxes_linear, return_variances=True),
            in_axes=(0, 0, batches_in_axes),
        ))

    fluxes_stack, variances_stack = solve_fn(initial_fluxes, images_data, batches)
    fluxes_np = np.asarray(fluxes_stack)
    var_np = np.asarray(variances_stack)

    # Pick the tile that contains the main source, then look up its slot.
    main_tile = find_main_tile(tile_records, sx_main, sy_main)
    slot = src_slot[main_tile].get(int(main_idx))
    if slot is None:
        return float("nan"), float("nan"), cutout.get("cwave_center")

    flux = float(fluxes_np[main_tile, slot])
    fvar = float(var_np[main_tile, slot])
    ferr = math.sqrt(max(fvar, 0.0))
    return flux, ferr, cutout.get("cwave_center")


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

def prepare_catalog(parquet_path: Path):
    tab = Table.read(parquet_path)
    e1 = np.asarray(tab["shape_e1"], dtype=np.float64)
    e2 = np.asarray(tab["shape_e2"], dtype=np.float64)
    e = np.hypot(e1, e2)
    ab = (1.0 - e) / (1.0 + e)
    phi = 0.5 * np.rad2deg(np.arctan2(e2, e1))
    phi = (phi + 180.0) % 180.0
    tab["shape_ab"] = ab
    tab["shape_phi"] = phi
    return tab


def find_main_source(tab: Table, ra_deg: float, dec_deg: float):
    sco = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    sep = sco.separation(target)
    return int(np.argmin(sep)), sco


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cutouts-dir", default="testdata/spherex_cutouts",
                   help="directory containing cutout_*.fits + summary.ecsv")
    p.add_argument("--catalog", default="tests/ls_newfield.parquet",
                   help="LS catalog parquet (output of fetch_ls_catalog_newfield.py)")
    p.add_argument("--ra", type=float, default=347.0925)
    p.add_argument("--dec", type=float, default=-2.1921)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--tile-halo", type=int, default=TILE_HALO)
    p.add_argument("--max-cutouts", type=int, default=None,
                   help="optionally process only the first N cutouts (debugging)")
    p.add_argument("--output", default="tests/test_jax_optimizer_spherex_batch_tiled.parquet")
    p.add_argument("--plot", default="tests/test_jax_optimizer_spherex_batch_tiled.png")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    cutouts_dir = Path(args.cutouts_dir)
    summary_path = cutouts_dir / "summary.ecsv"
    if summary_path.exists():
        summary = Table.read(summary_path)
        ok_mask = summary["status"] == "ok"
        summary_ok = summary[ok_mask]
    else:
        summary_ok = None

    pairs = discover_cutouts(cutouts_dir)
    if not pairs:
        raise SystemExit(f"No cutouts found under {cutouts_dir}")

    if summary_ok is not None:
        ok_indices = set(int(i) for i in summary_ok["cutout_index"])
        pairs = [(idx, p) for idx, p in pairs if idx in ok_indices]

    if args.max_cutouts:
        pairs = pairs[:args.max_cutouts]
    logger.info("Processing %d cutouts", len(pairs))

    tab = prepare_catalog(args.catalog)
    main_idx, sco_all = find_main_source(tab, args.ra, args.dec)
    main_co = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg)
    logger.info("Main source: ls_id=%s ra=%.6f dec=%.6f shape_r=%.3f sersic=%.3f",
                tab["ls_id"][main_idx],
                float(tab["ra"][main_idx]), float(tab["dec"][main_idx]),
                float(tab["shape_r"][main_idx]),
                float(tab["sersic"][main_idx]))

    results = Table(names=("cutout_index", "obs_id", "detector",
                            "central_wavelength", "flux", "flux_err"),
                    dtype=("i8", "U32", "i4", "f8", "f8", "f8"))

    # We let JIT cache compilations across calls — every cutout's tiles share
    # the same shapes (same tile_size+halo, same PSF size), so only the first
    # call pays the trace cost.
    solve_fn_cache = [None]
    t0 = time.time()
    for cutout_index, path in tqdm(pairs, desc="Cutouts"):
        try:
            cutout = read_cutout(path)
            flux, ferr, cwave = process_one_cutout(
                cutout, tab, sco_all, main_idx, main_co,
                profile_lookup_fn=SersicMixture.getProfile,
                psf_sampling=0.2, fixed_max_factor=5.0,
                tile_size=args.tile_size, halo=args.tile_halo,
                solve_fn=None,  # built per-call; jax.jit caches by signature
            )
        except Exception as exc:
            logger.exception("Cutout %d failed: %s", cutout_index, exc)
            continue
        obs_id = str(cutout["primary_header"].get("OBSID", ""))
        det = int(cutout["detector"])
        results.add_row((cutout_index, obs_id, det,
                          cwave if cwave is not None else float("nan"),
                          flux, ferr))
    logger.info("Total time: %.2f minutes", (time.time() - t0) / 60.0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.write(str(out), overwrite=True)
    logger.info("Wrote %s", out)

    if args.plot and len(results) > 0:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        m = np.isfinite(results["central_wavelength"]) & np.isfinite(results["flux"])
        ax.errorbar(np.asarray(results["central_wavelength"])[m],
                    np.asarray(results["flux"])[m],
                    yerr=np.asarray(results["flux_err"])[m],
                    fmt=".", color="tab:red", label="JAX tiled")
        ax.set_xlabel("Central wavelength (μm)")
        ax.set_ylabel("Flux (mJy)")
        ax.legend()
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=200)
        logger.info("Wrote %s", args.plot)


if __name__ == "__main__":
    main()
