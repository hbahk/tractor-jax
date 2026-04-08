import time
import re
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import jax
# jax.config.update("jax_enable_x64", True)
from photutils.background import Background2D, MedianBackground
from tractor_jax.jax.optimizer import (
    solve_fluxes_linear, extract_model_data_direct,
)
from tractor_jax.sersic import SersicMixture
from tqdm import tqdm, trange
from utils import get_nearest_psf_zone_index

logger = logging.getLogger(__name__)

THAW_SHAPE = False
THAW_POSITIONS = False

BKG_MODEL = "photutils"
# BKG_MODEL = "plane"
# BKG_MODEL = "none"
BKG_BOX_SIZE = 5
BKG_FILTER_SIZE = 3

# Pixel-area handling (mJy/sr -> mJy/pixel) via SIP WCS.
# Downsampling/grid options help speed while keeping SIP distortion.
PIXAREA_MODE = "sip_tangent"  # "sip_tangent","sip","constant"
PIXAREA_DOWNSAMPLE = 1  # integer >= 1
PIXAREA_USE_GRID = False  # if True, compute on coarse grid + interpolate
PIXAREA_GRID_STRIDE = 8  # grid step (pixels) when PIXAREA_USE_GRID is True
PIXAREA_CONST_SR = ((6.15 * u.arcsec)**2).to_value(u.sr)
IMG_SCALE = 1.0e9  # scale MJy -> mJy to improve numerical stability

# Bit definitions from FLAGS header
FLAG_BITS = {
    "TRANSIENT": 0,
    "OVERFLOW": 1,
    "SUR_ERROR": 2,
    "PHANTOM": 4,
    "REFERENCE": 5,
    "NONFUNC": 6,
    "DICHROIC": 7,
    "MISSING_DATA": 9,
    "HOT": 10,
    "COLD": 11,
    "FULLSAMPLE": 12,
    "PHANMISS": 14,
    "NONLINEAR": 15,
    "PERSIST": 17,
    "OUTLIER": 19,
    "SOURCE": 21,
}

MASK_FLAGS = [
    "SUR_ERROR",
    "PHANMISS",
    "NONFUNC",
    "MISSING_DATA",
    "HOT",
    "COLD",
    "PERSIST",
    "OUTLIER",
]

MASKBITS = 0
for name in MASK_FLAGS:
    MASKBITS |= (1 << FLAG_BITS[name])

def build_background_mask(flg, var, maskbits, source_bit):
    bad = (flg & maskbits) != 0
    source = (flg & source_bit) != 0
    valid_var = np.isfinite(var) & (var > 0)
    return bad | source | (~valid_var)

def fit_background_plane(img, bkg, flg, var, maskbits, source_bit, grid=None):
    """Fit order-1 2D polynomial to background pixels (weighted)."""
    if img.size == 0:
        return bkg

    # Build valid mask: exclude source + bad flags + non-finite/zero variance.
    mask = build_background_mask(flg, var, maskbits, source_bit)
    valid = ~mask

    if not np.any(valid):
        return bkg

    # Work on residual after existing background.
    z = (img - bkg)[valid].ravel()
    w = (1.0 / var[valid]).ravel()

    # Coordinates for design matrix.
    if grid is None:
        yy, xx = np.indices(img.shape)
    else:
        yy, xx = grid
    x = xx[valid].ravel().astype(np.float64)
    y = yy[valid].ravel().astype(np.float64)

    # Weighted least squares: z = a + b*x + c*y
    A = np.stack([np.ones_like(x), x, y], axis=1)
    Aw = A * w[:, None]
    AtAw = A.T @ Aw
    AtAz = A.T @ (z * w)
    try:
        coeff = np.linalg.solve(AtAw, AtAz)
    except np.linalg.LinAlgError:
        return bkg

    a, b, c = coeff
    plane = (a + b * xx + c * yy).astype(bkg.dtype, copy=False)
    return bkg + plane

def fit_background_photutils(img, bkg, flg, var, maskbits, source_bit, box_size, filter_size):
    """Fit background with photutils Background2D on residual image."""
    if img.size == 0:
        return bkg

    if img.shape[0] < box_size or img.shape[1] < box_size:
        return bkg

    mask = build_background_mask(flg, var, maskbits, source_bit)
    residual = img - bkg
    try:
        bkg2d = Background2D(
            residual,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            mask=mask,
            # bkg_estimator=MedianBackground(),
        )
    except Exception:
        return bkg

    return bkg + bkg2d.background.astype(bkg.dtype, copy=False)

def pad_array(arr, target_shape):
    pads = [(0, t - s) for s, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pads, mode='constant', constant_values=0)

def downsample_psf_oversample2(psf):
    """Downsample 2x while preserving center and total sum."""
    h, w = psf.shape
    cy, cx = h // 2, w // 2
    oh, ow = h // 2 + 1, w // 2 + 1
    ocy, ocx = oh // 2, ow // 2

    out = np.zeros((oh, ow), dtype=psf.dtype)

    # Quadrants (2x2 block averages)
    out[0:ocy, 0:ocx] = 0.25 * (
        psf[0:cy:2, 0:cx:2] + psf[1:cy:2, 0:cx:2]
        + psf[0:cy:2, 1:cx:2] + psf[1:cy:2, 1:cx:2]
    )
    out[0:ocy, ocx+1:ow] = 0.25 * (
        psf[0:cy:2, cx+1:w:2] + psf[1:cy:2, cx+1:w:2]
        + psf[0:cy:2, cx+2:w:2] + psf[1:cy:2, cx+2:w:2]
    )
    out[ocy+1:oh, 0:ocx] = 0.25 * (
        psf[cy+1:h:2, 0:cx:2] + psf[cy+2:h:2, 0:cx:2]
        + psf[cy+1:h:2, 1:cx:2] + psf[cy+2:h:2, 1:cx:2]
    )
    out[ocy+1:oh, ocx+1:ow] = 0.25 * (
        psf[cy+1:h:2, cx+1:w:2] + psf[cy+2:h:2, cx+1:w:2]
        + psf[cy+1:h:2, cx+2:w:2] + psf[cy+2:h:2, cx+2:w:2]
    )

    # Center row/column (1x2 or 2x1 averages)
    out[ocy, 0:ocx] = 0.5 * (psf[cy, 0:cx:2] + psf[cy, 1:cx:2])
    out[ocy, ocx+1:ow] = 0.5 * (psf[cy, cx+1:w:2] + psf[cy, cx+2:w:2])
    out[0:ocy, ocx] = 0.5 * (psf[0:cy:2, cx] + psf[1:cy:2, cx])
    out[ocy+1:oh, ocx] = 0.5 * (psf[cy+1:h:2, cx] + psf[cy+2:h:2, cx])

    # Center pixel
    out[ocy, ocx] = psf[cy, cx]

    # Preserve total flux
    total = psf.sum()
    out_sum = out.sum()
    if out_sum != 0:
        out *= total / out_sum

    return out

def _pixel_area_from_wcs_sip(wcs, shape, *, downsample=1, use_grid=True, grid_stride=8):
    """Compute per-pixel solid angle (sr) using SIP WCS.

    Options:
      - downsample: compute on a downsampled image shape
      - use_grid: compute on a coarse grid and interpolate
    """
    h, w = shape
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if grid_stride < 1:
        raise ValueError("grid_stride must be >= 1")

    dh = int(math.ceil(h / downsample))
    dw = int(math.ceil(w / downsample))

    # Too small to compute gradients; fall back to constant area.
    if dh < 2 or dw < 2:
        return np.full((h, w), PIXAREA_CONST_SR, dtype=np.float64)

    if use_grid:
        ys = np.arange(0, dh, grid_stride, dtype=np.float64)
        xs = np.arange(0, dw, grid_stride, dtype=np.float64)
        if ys[-1] != dh - 1:
            ys = np.append(ys, dh - 1)
        if xs[-1] != dw - 1:
            xs = np.append(xs, dw - 1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
    else:
        yy, xx = np.indices((dh, dw), dtype=np.float64)

    if yy.shape[0] < 2 or yy.shape[1] < 2:
        return np.full((h, w), PIXAREA_CONST_SR, dtype=np.float64)

    # Map to original pixel coordinates (center positions).
    xx_full = xx * downsample + 0.5 * (downsample - 1)
    yy_full = yy * downsample + 0.5 * (downsample - 1)

    ra, dec = wcs.pixel_to_world_values(xx_full, yy_full)  # deg
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # Unwrap RA to avoid 0/2pi discontinuity before differentiation.
    ra = np.unwrap(ra, axis=1)
    ra = np.unwrap(ra, axis=0)

    # Account for sampling step in gradient spacing.
    step = float(downsample * (grid_stride if use_grid else 1))
    ddec_dy, ddec_dx = np.gradient(dec, step, step)
    dra_dy, dra_dx = np.gradient(ra, step, step)

    omega = np.abs(
        (dra_dx * np.cos(dec)) * ddec_dy - (dra_dy * np.cos(dec)) * ddec_dx
    )

    if use_grid:
        # Bilinear interpolation onto downsampled full grid.
        yy_fullgrid, xx_fullgrid = np.indices((dh, dw), dtype=np.float64)
        y0 = np.floor((yy_fullgrid / grid_stride)).astype(int)
        x0 = np.floor((xx_fullgrid / grid_stride)).astype(int)
        y1 = np.clip(y0 + 1, 0, omega.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, omega.shape[1] - 1)
        wy = (yy_fullgrid / grid_stride) - y0
        wx = (xx_fullgrid / grid_stride) - x0

        omega00 = omega[y0, x0]
        omega01 = omega[y0, x1]
        omega10 = omega[y1, x0]
        omega11 = omega[y1, x1]
        omega = (
            (1 - wy) * (1 - wx) * omega00
            + (1 - wy) * wx * omega01
            + wy * (1 - wx) * omega10
            + wy * wx * omega11
        )

    if downsample == 1:
        return omega

    # Expand back to original shape by nearest-neighbor tiling.
    omega_full = np.repeat(np.repeat(omega, downsample, axis=0), downsample, axis=1)
    return omega_full[:h, :w]


def _pixel_area_from_wcs_sip_tangent_offsets(
    wcs,
    shape,
    *,
    downsample=1,
    grid_stride=8,
):
    """
    Compute per-pixel solid angle Omega [sr/pixel] using a SIP-including WCS
    via a *local Jacobian* estimated from tangent-plane offsets.

      - For each grid pixel center, build a local tangent plane at that center.
      - Compute spherical offsets (xi, eta) from the center to +/-x and +/-y
        neighbor points using SkyCoord.spherical_offsets_to().
      - Estimate Jacobian entries with finite differences.
      - Omega ~ |det d(xi,eta)/d(x,y)|, where (xi, eta) are in radians.

    Notes:
      - Uses a coarse grid for speed, then interpolates to full resolution.
      - Uses one-sided differences at the grid boundary as needed.
      - This avoids RA wrap issues and is generally more robust than differentiating RA/Dec.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS object (SIP is honored if present).
    shape : (H, W)
        Output image shape.
    downsample : int >= 1
        Additional coarsening factor (effective step = downsample*grid_stride pixels).
    grid_stride : int >= 1
        Grid step in pixels (after downsample factor is folded in).

    Returns
    -------
    omega_full : ndarray, shape (H, W), float64
        Solid angle per pixel [sr/pixel].
    """
    H, W = map(int, shape)
    if downsample < 1 or grid_stride < 1:
        raise ValueError("downsample and grid_stride must be >= 1")
    if H < 2 or W < 2:
        return np.full((H, W), PIXAREA_CONST_SR, dtype=np.float64)

    # Effective finite-difference step in *original* pixel units
    step = int(downsample) * int(grid_stride)
    if step < 1:
        step = 1

    # Grid pixel-center coordinates in original pixel space: centers are (i+0.5, j+0.5)
    xs = np.arange(0, W, step, dtype=np.float64) + 0.5
    ys = np.arange(0, H, step, dtype=np.float64) + 0.5

    # Ensure last grid point reaches the last pixel center
    if xs[-1] != (W - 0.5):
        xs = np.append(xs, W - 0.5)
    if ys[-1] != (H - 0.5):
        ys = np.append(ys, H - 0.5)

    if xs.size < 2 or ys.size < 2:
        return np.full((H, W), PIXAREA_CONST_SR, dtype=np.float64)

    # Mesh of grid pixel centers
    yy0, xx0 = np.meshgrid(ys, xs, indexing="ij")  # (Ny, Nx)

    # Neighbor pixel centers for finite differences (clamped at boundaries)
    x_min, x_max = 0.5, W - 0.5
    y_min, y_max = 0.5, H - 0.5

    xxp = np.clip(xx0 + step, x_min, x_max)
    xxm = np.clip(xx0 - step, x_min, x_max)
    yyp = np.clip(yy0 + step, y_min, y_max)
    yym = np.clip(yy0 - step, y_min, y_max)

    dxp = xxp - xx0
    dxm = xx0 - xxm
    dyp = yyp - yy0
    dym = yy0 - yym

    # WCS transform for center and neighbors (SIP included)
    # We keep everything as arrays for vectorization.
    ra0_deg, dec0_deg = wcs.pixel_to_world_values(xx0, yy0)
    rap_deg, decp_deg = wcs.pixel_to_world_values(xxp, yy0)
    ram_deg, decm_deg = wcs.pixel_to_world_values(xxm, yy0)
    rayp_deg, decyp_deg = wcs.pixel_to_world_values(xx0, yyp)
    raym_deg, decym_deg = wcs.pixel_to_world_values(xx0, yym)

    # Build SkyCoord objects (broadcasting works over arrays)
    c0 = SkyCoord(ra0_deg * u.deg, dec0_deg * u.deg, frame="icrs")
    cxp = SkyCoord(rap_deg * u.deg, decp_deg * u.deg, frame="icrs")
    cxm = SkyCoord(ram_deg * u.deg, decm_deg * u.deg, frame="icrs")
    cyp = SkyCoord(rayp_deg * u.deg, decyp_deg * u.deg, frame="icrs")
    cym = SkyCoord(raym_deg * u.deg, decym_deg * u.deg, frame="icrs")

    # Spherical offsets from center to neighbors in the local tangent plane
    # (dlon, dlat) are angles; convert to radians for a solid-angle Jacobian.
    xi_p,  eta_p  = c0.spherical_offsets_to(cxp)
    xi_m,  eta_m  = c0.spherical_offsets_to(cxm)
    xi_yp, eta_yp = c0.spherical_offsets_to(cyp)
    xi_ym, eta_ym = c0.spherical_offsets_to(cym)

    xi_p  = xi_p.to_value(u.rad)
    eta_p = eta_p.to_value(u.rad)
    xi_m  = xi_m.to_value(u.rad)
    eta_m = eta_m.to_value(u.rad)
    xi_yp  = xi_yp.to_value(u.rad)
    eta_yp = eta_yp.to_value(u.rad)
    xi_ym  = xi_ym.to_value(u.rad)
    eta_ym = eta_ym.to_value(u.rad)

    # Finite-difference derivatives with support for one-sided differences at boundaries.
    # For non-symmetric spacing (-dxm, +dxp), a reasonable linear estimate is:
    #   f'(0) ~ (f(+dxp) - f(-dxm)) / (dxp + dxm)
    # where f(-dxm) is the offset to the minus-point (typically negative in xi/eta).
    def two_sided_or_one_sided(fp, fm, dp, dm):
        out = np.empty_like(fp, dtype=np.float64)

        # Two-sided where both sides exist
        m2 = (dp > 0) & (dm > 0)
        out[m2] = (fp[m2] - fm[m2]) / (dp[m2] + dm[m2])

        # One-sided + where only + exists
        mp = (dp > 0) & ~(dm > 0)
        out[mp] = fp[mp] / dp[mp]

        # One-sided - where only - exists (note: fm is offset to the minus-point)
        mm = (dm > 0) & ~(dp > 0)
        out[mm] = -fm[mm] / dm[mm]

        # Degenerate (should be rare): set to 0
        mz = ~(m2 | mp | mm)
        out[mz] = 0.0
        return out

    dxi_dx  = two_sided_or_one_sided(xi_p,  xi_m,  dxp, dxm)
    deta_dx = two_sided_or_one_sided(eta_p, eta_m, dxp, dxm)
    dxi_dy  = two_sided_or_one_sided(xi_yp, xi_ym, dyp, dym)
    deta_dy = two_sided_or_one_sided(eta_yp, eta_ym, dyp, dym)

    # Solid angle per pixel ~ |det J| with J = d(xi,eta)/d(x,y)
    omega_grid = np.abs(dxi_dx * deta_dy - dxi_dy * deta_dx).astype(np.float64, copy=False)

    # If grid is already full resolution, return it
    if step == 1:
        return omega_grid

    # Interpolate from coarse grid (ys, xs) to full pixel-center grid using two-pass 1D interp.
    x_full = np.arange(W, dtype=np.float64) + 0.5
    y_full = np.arange(H, dtype=np.float64) + 0.5

    # 1) Interpolate along x for each coarse y
    omega_x = np.empty((ys.size, W), dtype=np.float64)
    for i in range(ys.size):
        omega_x[i, :] = np.interp(x_full, xs, omega_grid[i, :])

    # 2) Interpolate along y for each x
    omega_full = np.empty((H, W), dtype=np.float64)
    for j in range(W):
        omega_full[:, j] = np.interp(y_full, ys, omega_x[:, j])

    return omega_full

def _pixel_area_sr(wcs, shape):
    if PIXAREA_MODE == "constant":
        return np.full(shape, wcs.celestial.proj_plane_pixel_area().to_value(u.sr), dtype=np.float64)
    elif PIXAREA_MODE == "sip_tangent":
        return _pixel_area_from_wcs_sip_tangent_offsets(
            wcs,
            shape,
            downsample=PIXAREA_DOWNSAMPLE,
            grid_stride=PIXAREA_GRID_STRIDE,
        )
    elif PIXAREA_MODE == "sip":
        return _pixel_area_from_wcs_sip(
            wcs,
            shape,
            downsample=PIXAREA_DOWNSAMPLE,
            use_grid=PIXAREA_USE_GRID,
            grid_stride=PIXAREA_GRID_STRIDE,
        )
    raise ValueError(f"Invalid PIXAREA_MODE: {PIXAREA_MODE}")


def _fit_background_for_frame(args):
    """Worker for parallel background fitting."""
    img, bkg, flg, var, bkg_model, grid_cache_key = args
    if bkg_model == "photutils":
        return fit_background_photutils(
            img=img, bkg=bkg, flg=flg, var=var,
            maskbits=MASKBITS,
            source_bit=(1 << FLAG_BITS["SOURCE"]),
            box_size=BKG_BOX_SIZE, filter_size=BKG_FILTER_SIZE,
        )
    elif bkg_model == "plane":
        grid = np.indices(img.shape)
        return fit_background_plane(
            img=img, bkg=bkg, flg=flg, var=var,
            maskbits=MASKBITS,
            source_bit=(1 << FLAG_BITS["SOURCE"]),
            grid=grid,
        )
    return bkg


def test_jax_optimizer_spherex_batch_direct(idx_list, hdul, tab, solver="linear"):
    """
    Optimized version: builds JAX arrays directly, skipping Tractor objects.
    Uses direct linear solver by default.

    Args:
        idx_list: frame indices to process.
        hdul: already-opened FITS HDUList.
        tab: catalog table with ra, dec, shape_r, shape_ab, shape_phi, sersic.
        solver: "linear" or "cg".
    """
    start_time = time.time()
    cutout_info = Table(hdul[1].data)
    cutout_info["flux"] = np.full(len(cutout_info), np.nan)
    cutout_info["flux_err"] = np.full(len(cutout_info), np.nan)
    end_time = time.time()
    logger.info("Time to read the cutout info: %.6f seconds", end_time - start_time)

    # 1. Load frames and fit backgrounds (parallelized)
    max_h, max_w = 0, 0
    max_psf_h, max_psf_w = 0, 0

    logger.info("Loading frames and determining shapes...")
    bkg_args = []
    raw_frames = []

    for i in tqdm(idx_list, desc="Reading FITS"):
        img_idx = 2 + i * 6
        img = hdul[img_idx].data
        flg = hdul[img_idx + 1].data
        var = hdul[img_idx + 2].data
        bkg = hdul[img_idx + 3].data
        psf_cube = hdul[img_idx + 4].data
        psf_lookup = hdul[img_idx + 5].data
        hdr = hdul[img_idx].header

        max_h = max(max_h, img.shape[0])
        max_w = max(max_w, img.shape[1])

        gx, gy = cutout_info["x"][i], cutout_info["y"][i]
        zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
        zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
        psf_data = downsample_psf_oversample2(psf_cube[zidx])
        max_psf_h = max(max_psf_h, psf_data.shape[0])
        max_psf_w = max(max_psf_w, psf_data.shape[1])

        raw_frames.append({
            'img': img, 'flg': flg, 'var': var, 'bkg': bkg,
            'psf': psf_data, 'hdr': hdr, 'idx': i,
        })
        bkg_args.append((img, bkg, flg, var, BKG_MODEL, img.shape))

    # Parallel background fitting
    logger.info("Fitting backgrounds (parallel)...")
    start_time = time.time()
    if BKG_MODEL != "none":
        max_workers = min(len(bkg_args), os.cpu_count() or 1, 10)
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            fitted_bkgs = list(tqdm(
                executor.map(_fit_background_for_frame, bkg_args),
                total=len(bkg_args), desc="Background"
            ))
        for k, bkg_fitted in enumerate(fitted_bkgs):
            raw_frames[k]['bkg'] = bkg_fitted
    end_time = time.time()
    logger.info("Time to fit backgrounds: %.6f seconds", end_time - start_time)

    # 2. Preprocess frames into direct arrays
    logger.info("Preprocessing frames into JAX arrays...")
    start_time = time.time()

    psf_sampling = 0.2
    fixed_max_factor = 5.0
    fft_pad_h_lr = int(math.ceil(max_psf_h / fixed_max_factor))
    fft_pad_w_lr = int(math.ceil(max_psf_w / fixed_max_factor))
    padded_h = max_h + fft_pad_h_lr
    padded_w = max_w + fft_pad_w_lr
    fixed_target_shape = (
        int(round(padded_h * fixed_max_factor)),
        int(round(padded_w * fixed_max_factor)),
    )
    logger.info("Fixed target shape: %s, max_factor: %s", fixed_target_shape, fixed_max_factor)

    direct_frames = []
    for fr in raw_frames:
        img = fr['img']
        flg = fr['flg']
        var = fr['var']
        bkg = fr['bkg']
        hdr = fr['hdr']

        wcs = WCS(hdr)
        omega_sr = _pixel_area_sr(wcs, img.shape).astype(img.dtype, copy=False)
        img_scaled = img * omega_sr * IMG_SCALE
        bkg_scaled = bkg * omega_sr * IMG_SCALE
        var_scaled = var * (omega_sr ** 2) * (IMG_SCALE ** 2)

        invvar = 1.0 / var_scaled
        invvar[~np.isfinite(invvar)] = 0
        mask = flg & MASKBITS != 0
        invvar[mask] = 0

        data = pad_array(img_scaled - bkg_scaled, (padded_h, padded_w))
        data = np.where(np.isfinite(data), data, 0.0)
        invvar_pad = pad_array(invvar, (padded_h, padded_w))

        psf_pad = pad_array(fr['psf'], (max_psf_h, max_psf_w))

        direct_frames.append({
            'data': data,
            'invvar': invvar_pad,
            'psf': psf_pad,
            'wcs': wcs,
        })

    images_data, batches, initial_fluxes = extract_model_data_direct(
        direct_frames,
        tab,
        psf_sampling=psf_sampling,
        fit_background=True,
        fixed_target_shape=fixed_target_shape,
        fixed_max_factor=fixed_max_factor,
        profile_lookup_fn=SersicMixture.getProfile,
    )
    end_time = time.time()
    logger.info("Time to preprocess + extract: %.6f seconds", end_time - start_time)

    # 3. Run optimization
    logger.info("Running JAX optimization (solver=%s)...", solver)
    start_time = time.time()

    sample_batches = batches
    batches_in_axes = {}
    if "PointSource" in sample_batches:
        batches_in_axes["PointSource"] = {
            "flux_idx": 0, "pos_pix": 0, "mask": 0,
        }
    if "Galaxy" in sample_batches:
        batches_in_axes["Galaxy"] = {
            "flux_idx": 0, "pos_pix": 0, "wcs_cd_inv": 0,
            "mask": 0, "shapes": 0,
            "profile": {"amp": 0, "mean": 0, "var": 0},
        }
    if "Background" in sample_batches:
        batches_in_axes["Background"] = {"flux_idx": None}

    if solver == "linear":
        solve_fn = partial(solve_fluxes_linear, return_variances=True)
    else:
        from tractor_jax.jax.optimizer import solve_fluxes_core
        solve_fn = partial(solve_fluxes_core, return_variances=True, use_preconditioner=False)

    solve_vmap = jax.jit(jax.vmap(solve_fn, in_axes=(0, 0, batches_in_axes)))
    fluxes_stack, variances_stack = solve_vmap(initial_fluxes, images_data, batches)

    end_time = time.time()
    logger.info("Time to optimize fluxes: %.6f seconds", end_time - start_time)

    # 4. Diagnostics (simplified)
    f_np = np.array(fluxes_stack)
    v_np = np.array(variances_stack)

    nan_flux = np.isnan(f_np)
    inf_flux = np.isinf(f_np)
    nan_var = np.isnan(v_np)
    inf_var = np.isinf(v_np)
    neg_flux = f_np < 0.0

    logger.info("=== Diagnostics ===")
    logger.info("fluxes   — NaN: %d  Inf: %d  Negative: %d  (out of %d)",
                nan_flux.sum(), inf_flux.sum(), neg_flux.sum(), f_np.size)
    logger.info("variances — NaN: %d  Inf: %d",
                nan_var.sum(), inf_var.sum())

    # Find main source
    gra, gdec = 258.2084186 * u.deg, 64.0529535 * u.deg
    gco = SkyCoord(ra=gra, dec=gdec)
    sc = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    sep = sc.separation(gco)
    main_idx = np.argmin(sep)

    flux = f_np[:, main_idx]
    ferr = np.sqrt(np.clip(v_np[:, main_idx], 0, None))

    logger.info("Main flux sample: %s", flux[:5])

    for k, i in enumerate(idx_list):
        cutout_info["flux"][i] = flux[k]
        cutout_info["flux_err"][i] = ferr[k]

    return cutout_info


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start_time = time.time()

    if os.path.exists("tests/testphot.fits"):
        hdul = fits.open("tests/testphot.fits")
        cutout_info = Table(hdul[1].data)
        cutout_info["flux"] = np.full(len(cutout_info), np.nan)
        cutout_info["flux_err"] = np.full(len(cutout_info), np.nan)

        tab = Table.read("tests/ls_testgal.parquet")
        tco = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
        e1, e2 = tab["shape_e1"], tab["shape_e2"]
        e = np.hypot(e1, e2)
        ab = (1 - e) / (1 + e)
        phi = 0.5 * np.rad2deg(np.arctan2(e2, e1))
        phi = (phi + 180.0) % 180.0
        tab["shape_phi"] = phi
        tab["shape_ab"] = ab

        test_index = np.arange(len(cutout_info))
        batch_size = 500
        solver = "linear"  # "linear" or "cg"

        for i in range(0, len(test_index), batch_size):
            batch_index = test_index[i:i+batch_size]
            batch_cutout_info = test_jax_optimizer_spherex_batch_direct(
                batch_index, hdul, tab, solver=solver
            )
            cutout_info["flux"][batch_index] = batch_cutout_info["flux"][batch_index]
            cutout_info["flux_err"][batch_index] = batch_cutout_info["flux_err"][batch_index]

        hdul.close()
        cutout_info.write("tests/test_jax_optimizer_spherex_batch.parquet", overwrite=True)

        if os.path.exists("/data1/hbahk/spherex-cluster/codes/realworld/specphot_results_testgal_a2255_b.parquet"):
            cpu_result = Table.read("/data1/hbahk/spherex-cluster/codes/realworld/specphot_results_testgal_a2255_b.parquet")

            wave = cpu_result["central_wavelength"]
            flux = cpu_result["flux"]
            ferr = cpu_result["flux_err"]

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.errorbar(wave, flux, ferr, fmt=".", label="flux (CPU)", color="tab:blue")
            ax.errorbar(cutout_info["central_wavelength"], cutout_info["flux"], cutout_info["flux_err"], fmt=".", label="flux (JAX Batched)", color="tab:red")
            ax.legend()
            fig.savefig("tests/test_jax_optimizer_spherex_batch_comparison.png", dpi=300, bbox_inches='tight')
        end_time = time.time()
        logger.info("Time to run the test: %.6f minutes", (end_time - start_time) / 60)
    else:
        logger.info("Test data not found. Skipping execution.")
