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
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from photutils.background import Background2D, MedianBackground
from tractor import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor.brightness import Flux
from tractor.wcs import PixPos, AstropyWCS, RaDecPos, AffineWCS
from tractor.jax.optimizer import JaxOptimizer, extract_model_data, solve_fluxes_core
from tractor.psf import PixelizedPSF
import tractor
from tractor import ConstantSky, Flux, LinearPhotoCal, NullWCS, PixPos, PointSource, RaDecPos
from tractor.galaxy import GalaxyShape, JaxGalaxy
from tractor.utils import MogParams
from tractor.sersic import SersicIndex, SersicGalaxy, SersicMixture
from tqdm import tqdm, trange
from utils import get_nearest_psf_zone_index, sky_pa_to_pixel_pa, SPHERExSersicGalaxy, SPHERExSersicMixture

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

def pad_mog(mog_params, max_K):
    # mog_params is tuple/object with amp, mean, var
    # Input is MogParams (from tractor.utils) or simple object
    # Here we assume it's MogParams which has amp, mean, var as arrays (or lists)

    amp = np.array(mog_params.amp)
    mean = np.array(mog_params.mean)
    var = np.array(mog_params.var)

    K = len(amp)
    if K == max_K:
        return amp, mean, var

    pad = max_K - K
    new_amp = np.pad(amp, (0, pad))
    new_mean = np.pad(mean, ((0, pad), (0, 0)))

    # Var padding: identity
    new_var = np.pad(var, ((0, pad), (0, 0), (0, 0)))
    if pad > 0:
        # Broadcasting identity to (pad, 2, 2)
        new_var[K:] = np.eye(2)

    return new_amp, new_mean, new_var

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

def _build_tractor_for_frame(
    frame,
    tab,
    max_h,
    max_w,
    max_psf_h,
    max_psf_w,
    max_mog_K,
    thaw_shape,
    thaw_positions,
    maskbits,
):
    img = frame["img"]
    flg = frame["flg"]
    var = frame["var"]
    bkg = frame["bkg"]
    psf_data = frame["psf"]
    hdr = frame["hdr"]
    idx = frame["idx"]

    # Padding
    img_padded = pad_array(img, (max_h, max_w))
    bkg_padded = pad_array(bkg, (max_h, max_w))

    invvar = 1 / var
    mask = flg & maskbits != 0
    invvar[mask] = 0
    invvar_padded = pad_array(invvar, (max_h, max_w))

    psf_padded = pad_array(psf_data, (max_psf_h, max_psf_w))
    psf_tractor = PixelizedPSF(psf_padded, sampling=0.2)

    # WCS
    wcs = WCS(hdr)
    wcs_tractor = AstropyWCS(wcs)

    # Affine WCS approximates the wcs using original image center.
    orig_cx, orig_cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    cd = wcs_tractor.cdAtPixel(orig_cx, orig_cy)
    center_sky = wcs_tractor.pixelToPosition(orig_cx, orig_cy)
    crval = [center_sky.ra, center_sky.dec]
    crpix = [orig_cx, orig_cy]
    affine_wcs = AffineWCS(crpix, crval, cd)

    # Create Image
    tim = tractor.Image(
        data=img_padded - bkg_padded,
        inverr=np.sqrt(invvar_padded),
        psf=psf_tractor,
        # wcs=affine_wcs,
        wcs=NullWCS(pixscale=6.15),
        photocal=LinearPhotoCal(1.0),
        sky=ConstantSky(0.0),
    )
    tim.freezeAllRecursive()
    tim.thawPathsTo("sky")

    # Sources
    frame_sources = []
    stab = tab
    sco = SkyCoord(ra=stab["ra"], dec=stab["dec"], unit="deg")
    pxs, pys = wcs.world_to_pixel(sco)
    rng = np.random.default_rng(idx)
    for row, px, py in zip(stab, pxs, pys):
        # _flux = Flux(rng.uniform(high=1))
        ix = int(round(px))
        iy = int(round(py))
        if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
            init_flux_val = img[iy, ix] - bkg[iy, ix]
        else:
            init_flux_val = 0.0
        _flux = Flux(init_flux_val)

        # Use original WCS for position (valid for padded image since origin preserved)
        # px, py = wcs_tractor.positionToPixel(RaDecPos(row["ra"], row["dec"]))
        # px, py = row["x"], row["y"]

        if row["shape_r"] == 0:
            _src = PointSource(PixPos(px, py), _flux)
        else:
            phi_img = sky_pa_to_pixel_pa(
                wcs, row["ra"], row["dec"], row["shape_phi"], d_arcsec=1.0, y_down=False
            )
            shape = GalaxyShape(row["shape_r"], row["shape_ab"], phi_img)

            profile_mog = SPHERExSersicMixture.getProfile(row["sersic"])
            amp, mean, var = pad_mog(profile_mog, max_mog_K)

            prof_params = MogParams(np.array(amp), np.array(mean), np.array(var))

            _src = JaxGalaxy(
                PixPos(px, py),
                _flux,
                shape,
                prof_params,
            )

        _src.freezeAllRecursive()
        _src.thawParam("brightness")
        if row["shape_r"] > 0 and thaw_shape:
            _src.thawPathsTo("shape")
        if thaw_positions:
            _src.thawPathsTo("pos")

        frame_sources.append(_src)

    optimizer = JaxOptimizer()
    return Tractor([tim], frame_sources, optimizer=optimizer)

def test_jax_optimizer_spherex_batch(idx_list):
    start_time = time.time()
    hdul = fits.open("tests/testphot.fits")
    end_time = time.time()
    logger.info("Time to open the output file: %.6f seconds", end_time - start_time)

    start_time = time.time()
    cutout_info = Table(hdul[1].data)
    cutout_info["flux"] = np.full(len(cutout_info), np.nan)
    cutout_info["flux_err"] = np.full(len(cutout_info), np.nan)
    end_time = time.time()
    logger.info("Time to read the cutout info: %.6f seconds", end_time - start_time)

    tab = Table.read("tests/ls_testgal.parquet")
    tco = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    
    e1, e2 = tab["shape_e1"], tab["shape_e2"]
    e = np.hypot(e1, e2)
    ab = (1 - e) / (1 + e)  # axis ratio = b/a
    phi = 0.5 * np.rad2deg(np.arctan2(e2, e1))
    phi = (phi + 180.0) % 180.0
    tab["shape_phi"] = phi
    tab["shape_ab"] = ab

    nframes = (len(hdul) - 2) // 6

    # 1. First Pass: Collect data and determine Max Shapes
    frames_data = []
    max_h, max_w = 0, 0
    max_psf_h, max_psf_w = 0, 0
    max_mog_K = 0
    grid_cache = {}

    logger.info("Loading frames and determining shapes...")
    for i in tqdm(idx_list):
        img_idx = 2 + i * 6
        flg_idx = img_idx + 1
        var_idx = img_idx + 2
        bkg_idx = img_idx + 3
        psf_idx = img_idx + 4
        psf_lookup_idx = img_idx + 5
        
        img = hdul[img_idx].data
        max_h = max(max_h, img.shape[0])
        max_w = max(max_w, img.shape[1])
        
        flg = hdul[flg_idx].data
        var = hdul[var_idx].data
        bkg = hdul[bkg_idx].data

        # Fit background on non-source, non-bad pixels.
        if BKG_MODEL == "photutils":
            bkg = fit_background_photutils(
                img=img,
                bkg=bkg,
                flg=flg,
                var=var,
                maskbits=MASKBITS,
                source_bit=(1 << FLAG_BITS["SOURCE"]),
                box_size=BKG_BOX_SIZE,
                filter_size=BKG_FILTER_SIZE,
            )
        elif BKG_MODEL == "plane":
            shape_key = img.shape
            if shape_key not in grid_cache:
                grid_cache[shape_key] = np.indices(shape_key)
            bkg = fit_background_plane(
                img=img,
                bkg=bkg,
                flg=flg,
                var=var,
                maskbits=MASKBITS,
                source_bit=(1 << FLAG_BITS["SOURCE"]),
                grid=grid_cache[shape_key],
            )
        elif BKG_MODEL == "none":
            pass
        else:
            raise ValueError(f"Invalid background model: {BKG_MODEL}")

        psf_cube = hdul[psf_idx].data
        psf_lookup = hdul[psf_lookup_idx].data

        gx, gy = cutout_info["x"][i], cutout_info["y"][i]
        zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
        zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
        psf_data = downsample_psf_oversample2(psf_cube[zidx])
        max_psf_h = max(max_psf_h, psf_data.shape[0])
        max_psf_w = max(max_psf_w, psf_data.shape[1])

        hdr = hdul[img_idx].header

        # Convert mJy/sr -> mJy/pixel using SIP WCS pixel area.
        wcs = WCS(hdr)
        omega_sr = _pixel_area_sr(wcs, img.shape).astype(img.dtype, copy=False)
        img = img * omega_sr * IMG_SCALE
        bkg = bkg * omega_sr * IMG_SCALE
        var = var * (omega_sr ** 2) * (IMG_SCALE ** 2)

        frames_data.append({
            'img': img,
            'flg': flg,
            'var': var,
            'bkg': bkg,
            'psf': psf_data,
            'hdr': hdr,
            'idx': i
        })

    # Determine max MoG K from catalog
    for row in tab:
        if row["shape_r"] > 0:
            profile_mog = SPHERExSersicMixture.getProfile(row["sersic"])
            max_mog_K = max(max_mog_K, len(profile_mog.amp))

    logger.info("Max Image Shape: %sx%s", max_h, max_w)
    logger.info("Max PSF Shape: %sx%s", max_psf_h, max_psf_w)
    logger.info("Max MoG K: %s", max_mog_K)

    # Compute a fixed target shape so all frames stack cleanly.
    # Sampling is fixed at 0.2 in _build_tractor_for_frame -> max_factor = 1 / 0.2 = 5.
    fixed_max_factor = 5.0
    fft_pad_h_lr = int(math.ceil(max_psf_h / fixed_max_factor))
    fft_pad_w_lr = int(math.ceil(max_psf_w / fixed_max_factor))
    padded_h = max_h + fft_pad_h_lr
    padded_w = max_w + fft_pad_w_lr
    fixed_target_shape = (
        int(round(padded_h * fixed_max_factor)),
        int(round(padded_w * fixed_max_factor)),
    )

    # 2. Second Pass: Create Tractors with Padding
    tractor_list = []

    logger.info("Constructing Tractors...")
    if frames_data:
        # max_workers = min(len(frames_data), os.cpu_count() or 1)
        max_workers = 10
        mp_context = mp.get_context("spawn")
        build_fn = partial(
            _build_tractor_for_frame,
            tab=tab,
            max_h=max_h,
            max_w=max_w,
            max_psf_h=max_psf_h,
            max_psf_w=max_psf_w,
            max_mog_K=max_mog_K,
            thaw_shape=THAW_SHAPE,
            thaw_positions=THAW_POSITIONS,
            maskbits=MASKBITS,
        )
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        ) as executor:
            tractor_list = list(
                tqdm(executor.map(build_fn, frames_data), total=len(frames_data))
            )

    logger.info("Precomputing model data...")
    start_time = time.time()

    images_data_list = []
    batches_list = []
    fluxes_list = []

    for trac in tractor_list:
        images_data, batches, initial_fluxes = extract_model_data(
            trac,
            oversample_rendering=True,
            fit_background=True,
            fixed_target_shape=fixed_target_shape,
            fixed_max_factor=fixed_max_factor,
        )
        images_data_list.append(images_data)
        batches_list.append(batches)
        fluxes_list.append(initial_fluxes)

    end_time = time.time()
    logger.info("Time to precompute: %.6f seconds", end_time - start_time)

    logger.info("Stacking model data...")
    start_time = time.time()
    images_data_batched = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x), *images_data_list
    )
    batches_batched = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x), *batches_list
    )
    fluxes_batched = jnp.stack(fluxes_list)
    end_time = time.time()
    logger.info("Time to stack: %.6f seconds", end_time - start_time)

    logger.info("Running JAX optimization...")
    start_time = time.time()

    sample_batches = batches_list[0] if batches_list else {}
    batches_in_axes = {}
    if "PointSource" in sample_batches:
        batches_in_axes["PointSource"] = {
            "flux_idx": 0,
            "pos_pix": 0,
            "mask": 0,
        }
    if "Galaxy" in sample_batches:
        batches_in_axes["Galaxy"] = {
            "flux_idx": 0,
            "pos_pix": 0,
            "wcs_cd_inv": 0,
            "mask": 0,
            "shapes": 0,
            "profile": {
                "amp": 0,
                "mean": 0,
                "var": 0,
            },
        }
    if "Background" in sample_batches:
        batches_in_axes["Background"] = {
            "flux_idx": None
        }

    solve_images_vmap = jax.vmap(
        partial(solve_fluxes_core, return_variances=True, use_preconditioner=False),
        in_axes=(0, 0, batches_in_axes)
    )
    solve_tractors_vmap = jax.vmap(
        solve_images_vmap,
        in_axes=(0, 0, 0)
    )

    fluxes_stack, variances_stack = jax.jit(solve_tractors_vmap)(
        fluxes_batched, images_data_batched, batches_batched
    )
    end_time = time.time()
    logger.info("Time to optimize fluxes: %.6f seconds", end_time - start_time)

    # If each tractor has a single image, drop the image axis.
    if fluxes_stack.ndim == 3:
        fluxes_stack = fluxes_stack[:, 0]
        variances_stack = variances_stack[:, 0]

    gra, gdec = 258.2084186 * u.deg, 64.0529535 * u.deg
    gco = SkyCoord(ra=gra, dec=gdec)
    sc = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    sep = sc.separation(gco)
    main_idx = np.argmin(sep)

    flux = np.array(fluxes_stack)[:, main_idx]
    ferr = np.sqrt(np.array(variances_stack)[:, main_idx])

    logger.info("Final Flux Sample: %s", flux[:5])

    for i in range(len(idx_list)):
        cutout_info["flux"][idx_list[i]] = flux[i]
        cutout_info["flux_err"][idx_list[i]] = ferr[i]
        
    return cutout_info

if __name__ == "__main__":
    from tractor.jax.tree import register_pytree_nodes
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        register_pytree_nodes()
    except ValueError:
        pass

    start_time = time.time()

    # test_index = np.arange(0, 3000, 30)
    # Check if files exist to avoid error
    import os
    if os.path.exists("tests/testphot.fits"):
        hdul = fits.open("tests/testphot.fits")
        cutout_info = Table(hdul[1].data)
        cutout_info["flux"] = np.full(len(cutout_info), np.nan)
        cutout_info["flux_err"] = np.full(len(cutout_info), np.nan)
        test_index = np.arange(len(cutout_info))
        batch_size = 100
        # for i in range(0, len(test_index), batch_size):
        for i in range(1000, 2000, batch_size):
            batch_index = test_index[i:i+batch_size]
            batch_cutout_info = test_jax_optimizer_spherex_batch(batch_index)
            cutout_info["flux"][batch_index] = batch_cutout_info["flux"][batch_index]
            cutout_info["flux_err"][batch_index] = batch_cutout_info["flux_err"][batch_index]

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
