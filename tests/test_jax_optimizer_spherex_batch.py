import time
import re

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
from tractor import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor.brightness import Flux
from tractor.wcs import PixPos, AstropyWCS, RaDecPos, AffineWCS
from tractor.jax.optimizer import JaxOptimizer, extract_model_data, optimize_fluxes, render_image
from tractor.psf import PixelizedPSF
import tractor
from tractor import ConstantSky, Flux, LinearPhotoCal, NullWCS, PixPos, PointSource, RaDecPos
from tractor.galaxy import GalaxyShape, JaxGalaxy
from tractor.utils import MogParams
from tractor.sersic import SersicIndex, SersicGalaxy, SersicMixture
from tqdm import tqdm, trange
from utils import get_nearest_psf_zone_index, sky_pa_to_pixel_pa, SPHERExSersicGalaxy, SPHERExSersicMixture

THAW_SHAPE = False
THAW_POSITIONS = False

PIX_SR = ((6.15 * u.arcsec)**2).to_value(u.sr)

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

def test_jax_optimizer_spherex_batch(idx_list):
    start_time = time.time()
    hdul = fits.open("tests/testphot.fits")
    end_time = time.time()
    print(f"Time to open the output file: {end_time - start_time} seconds")

    start_time = time.time()
    cutout_info = Table(hdul[1].data)
    cutout_info["flux"] = np.full(len(cutout_info), np.nan)
    cutout_info["flux_err"] = np.full(len(cutout_info), np.nan)
    end_time = time.time()
    print(f"Time to read the cutout info: {end_time - start_time} seconds")

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

    print("Loading frames and determining shapes...")
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

        psf_cube = hdul[psf_idx].data
        psf_lookup = hdul[psf_lookup_idx].data

        gx, gy = cutout_info["x"][i], cutout_info["y"][i]
        zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
        zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
        psf_data = psf_cube[zidx]
        max_psf_h = max(max_psf_h, psf_data.shape[0])
        max_psf_w = max(max_psf_w, psf_data.shape[1])

        hdr = hdul[img_idx].header

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

    print(f"Max Image Shape: {max_h}x{max_w}")
    print(f"Max PSF Shape: {max_psf_h}x{max_psf_w}")
    print(f"Max MoG K: {max_mog_K}")

    # 2. Second Pass: Create Tractors with Padding
    tractor_list = []

    print("Constructing Tractors...")
    for frame in tqdm(frames_data):
        img = frame['img']
        flg = frame['flg']
        var = frame['var']
        bkg = frame['bkg']
        psf_data = frame['psf']
        hdr = frame['hdr']
        i = frame['idx']

        # Padding
        img_padded = pad_array(img, (max_h, max_w))
        bkg_padded = pad_array(bkg, (max_h, max_w))

        invvar = 1 / var
        mask = flg & MASKBITS != 0
        invvar[mask] = 0
        invvar_padded = pad_array(invvar, (max_h, max_w))

        psf_padded = pad_array(psf_data, (max_psf_h, max_psf_w))
        psf_tractor = PixelizedPSF(psf_padded, sampling=0.1)

        # WCS
        wcs = WCS(hdr)
        wcs_tractor = AstropyWCS(wcs)

        cx, cy = max_w / 2.0, max_h / 2.0 # Use padded center
        # Or should we use original center?
        # Affine WCS approximates the wcs.
        # Since we padded right/bottom, the WCS origin (0,0) is same.
        # But for linearization, better use actual image center.
        # Original center:
        orig_cx, orig_cy = img.shape[1]/2.0, img.shape[0]/2.0
        
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
            wcs=affine_wcs,
            photocal=LinearPhotoCal(1.0),
            sky=ConstantSky(0.0),
        )
        tim.freezeAllRecursive()
        tim.thawPathsTo("sky")
        
        # Sources
        frame_sources = []
        stab = tab
        for row in stab:
            _flux = Flux(np.random.uniform(high=1))
            
            # Use original WCS for position (valid for padded image since origin preserved)
            px, py = wcs_tractor.positionToPixel(RaDecPos(row["ra"], row["dec"]))
            
            if row["shape_r"] == 0:
                _src = PointSource(PixPos(px, py), _flux)
            else:
                phi_img = sky_pa_to_pixel_pa(wcs, row["ra"], row["dec"], row["shape_phi"], d_arcsec=1.0, y_down=False)
                shape = GalaxyShape(row["shape_r"], row["shape_ab"], phi_img)

                profile_mog = SPHERExSersicMixture.getProfile(row["sersic"])
                amp, mean, var = pad_mog(profile_mog, max_mog_K)

                prof_params = MogParams(jnp.array(amp), jnp.array(mean), jnp.array(var))

                _src = JaxGalaxy(
                    PixPos(px, py),
                    _flux,
                    shape,
                    prof_params
                )

            _src.freezeAllRecursive()
            _src.thawParam("brightness")
            if row["shape_r"] > 0 and THAW_SHAPE:
                _src.thawPathsTo("shape")
            if THAW_POSITIONS:
                _src.thawPathsTo("pos")

            frame_sources.append(_src)

        trac = Tractor([tim], frame_sources)
        tractor_list.append(trac)

    print("Stacking Tractor objects...")
    start_time = time.time()
    batched_tractor = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *tractor_list)
    end_time = time.time()
    print(f"Time to stack: {end_time - start_time} seconds")
    
    print("Running JAX optimization...")
    start_time = time.time()

    def run_opt(trac):
        return optimize_fluxes(
            trac,
            return_variances=True,
            fit_background=True,
            oversample_rendering=True,
            vmap_images=False,
            update_catalog=False
        )

    vmap_opt = jax.jit(jax.vmap(run_opt))
    results = vmap_opt(batched_tractor)
    end_time = time.time()
    print(f"Time to optimize fluxes: {end_time - start_time} seconds")

    fluxes_stack, variances_stack = results[0]

    gra, gdec = 258.2084186 * u.deg, 64.0529535 * u.deg
    gco = SkyCoord(ra=gra, dec=gdec)
    sc = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    sep = sc.separation(gco)
    main_idx = np.argmin(sep)

    flux = np.array(fluxes_stack)[:, main_idx] * PIX_SR * 1.0e9
    ferr = np.sqrt(np.array(variances_stack)[:, main_idx]) * PIX_SR * 1.0e9

    print(f"Final Flux Sample: {flux[:5]}")

    for i in range(len(idx_list)):
        cutout_info["flux"][idx_list[i]] = flux[i]
        cutout_info["flux_err"][idx_list[i]] = ferr[i]
        
    return cutout_info

if __name__ == "__main__":
    from tractor.jax.tree import register_pytree_nodes
    try:
        register_pytree_nodes()
    except ValueError:
        pass

    test_index = np.arange(0, 3000, 100)
    # Check if files exist to avoid error
    import os
    if os.path.exists("tests/testphot.fits"):
        cutout_info = test_jax_optimizer_spherex_batch(test_index)
        cutout_info.write("tests/test_jax_optimizer_spherex_batch.parquet", overwrite=True)

        if os.path.exists("/data1/hbahk/spherex-cluster/codes/realworld/specphot_results_testgal_a2255_b.parquet"):
            cpu_result = Table.read("/data1/hbahk/spherex-cluster/codes/realworld/specphot_results_testgal_a2255_b.parquet")

            wave = cpu_result["central_wavelength"]
            flux = cpu_result["flux"]
            ferr = cpu_result["flux_err"]

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.errorbar(wave, flux, ferr, fmt="o", label="flux (CPU)")
            ax.errorbar(cutout_info["central_wavelength"], cutout_info["flux"], cutout_info["flux_err"], fmt="o", label="flux (JAX Batched)")
            ax.legend()
            fig.savefig("tests/test_jax_optimizer_spherex_batch_comparison.png", dpi=300, bbox_inches='tight')
    else:
        print("Test data not found. Skipping execution.")
