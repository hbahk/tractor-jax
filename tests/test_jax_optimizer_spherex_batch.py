import concurrent
import time
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tractor import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor.brightness import Flux
from tractor.wcs import PixPos, AstropyWCS, RaDecPos
from tractor.jax.optimizer import JaxOptimizer, extract_model_data, optimize_fluxes, render_image
from tractor.psf import PixelizedPSF
import tractor
from tractor import ConstantSky, Flux, LinearPhotoCal, NullWCS, PixPos, PointSource, RaDecPos
from tractor.galaxy import GalaxyShape
from tractor.sersic import SersicIndex, SersicGalaxy, SersicMixture
from tqdm import tqdm, trange
from utils import get_nearest_psf_zone_index, sky_pa_to_pixel_pa, SPHERExSersicGalaxy
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
    # phi = -np.rad2deg(np.arctan2(e2, e1) / 2)
    phi = 0.5 * np.rad2deg(np.arctan2(e2, e1))
    phi = (phi + 180.0) % 180.0
    tab["shape_phi"] = phi
    tab["shape_ab"] = ab

    nframes = (len(hdul) - 2) // 6
    tims = []

    # for i in trange(nframes):
    for i in tqdm(idx_list):
        img_idx = 2 + i * 6
        flg_idx = img_idx + 1
        var_idx = img_idx + 2
        bkg_idx = img_idx + 3
        psf_idx = img_idx + 4
        psf_lookup_idx = img_idx + 5
        
        img = hdul[img_idx].data
        hdr = hdul[img_idx].header
        wcs = WCS(hdr)
        
        wcs_tractor = AstropyWCS(wcs)

        flg = hdul[flg_idx].data
        var = hdul[var_idx].data
        bkg = hdul[bkg_idx].data

        invvar = 1 / var
        mask = flg & MASKBITS != 0
        invvar[mask] = 0

        psf_cube = hdul[psf_idx].data
        psf_lookup = hdul[psf_lookup_idx].data

        gx, gy = cutout_info["x"][i], cutout_info["y"][i]
        zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
        zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
        psf = psf_cube[zidx]

        psf_tractor = PixelizedPSF(psf, sampling=0.1)

        # get the pixel coordinates of the target
        tx, ty = wcs.world_to_pixel(tco)
        tab["x"], tab["y"] = tx, ty

        # tinside = (tx > -0.5) & (tx < img.shape[1]+0.5) & (ty > -0.5) & (ty < img.shape[0]+0.5)

        gra, gdec = 258.2084186 * u.deg, 64.0529535 * u.deg
        gco = SkyCoord(ra=gra, dec=gdec)

        stab = tab#[tinside]
        sc = SkyCoord(ra=stab["ra"], dec=stab["dec"], unit="deg")
        sep = sc.separation(gco)
        main_idx = np.argmin(sep)
        
        tim = tractor.Image(
            data=img - bkg,
            inverr=np.sqrt(invvar),
            psf=psf_tractor,
            # wcs=NullWCS(pixscale=6.15),
            wcs=wcs_tractor,
            photocal=LinearPhotoCal(1.0),
            sky=ConstantSky(0.0),
        )

        tim.freezeAllRecursive()
        tim.thawPathsTo("sky")
        
        tims.append(tim)
        
    tractor_source_list = []
    for row in stab:
        _flux = Flux(np.random.uniform(high=1))
        if row["shape_r"] == 0:
            # _src = PointSource(PixPos(row["x"], row["y"]), _flux)
            _src = PointSource(RaDecPos(row["ra"], row["dec"]), _flux)
        else:
            phi_img = sky_pa_to_pixel_pa(wcs, row["ra"], row["dec"], row["shape_phi"], d_arcsec=1.0, y_down=False)
            
            _src = SPHERExSersicGalaxy(
                # PixPos(row["x"], row["y"]),
                RaDecPos(row["ra"], row["dec"]),
                _flux,
                GalaxyShape(row["shape_r"], row["shape_ab"], phi_img),
                SersicIndex(row["sersic"]),
            )

        _src.freezeAllRecursive()
        _src.thawParam("brightness")

        if row["shape_r"] > 0:
            if THAW_SHAPE:
                _src.thawPathsTo("re")
                _src.thawPathsTo("ab")
                _src.thawPathsTo("phi")

        if THAW_POSITIONS:
            _src.thawPathsTo("x")
            _src.thawPathsTo("y")

        tractor_source_list.append(_src)
            
    trac_spherex = tractor.Tractor(tims, tractor_source_list)
    
    start_time = time.time()
    res = optimize_fluxes(trac_spherex, return_variances=True, fit_background=True, oversample_rendering=True)
    end_time = time.time()
    print(f"Time to optimize fluxes: {end_time - start_time} seconds")
    flux = np.array(res)[:, 0, main_idx] * PIX_SR * 1.0e9
    ferr = np.sqrt(np.array(res)[:, 1, main_idx]) * PIX_SR * 1.0e9
    print(f"Final Flux: {flux}, Flux Error: {ferr}")
    for i in range(len(idx_list)):
        cutout_info["flux"][idx_list[i]] = flux[i]
        cutout_info["flux_err"][idx_list[i]] = ferr[i]
        
    return cutout_info

if __name__ == "__main__":
    
    # test_index = np.arange(0, 3000, 60)
    test_index = np.arange(0, 3000, 100)
    cutout_info = test_jax_optimizer_spherex_batch(test_index)
    cutout_info.write("tests/test_jax_optimizer_spherex_batch.parquet", overwrite=True)
    
    cpu_result = Table.read("/data1/hbahk/spherex-cluster/codes/realworld/specphot_results_testgal_a2255_b.parquet")

    wave = cpu_result["central_wavelength"]
    flux = cpu_result["flux"]
    ferr = cpu_result["flux_err"]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(wave, flux, ferr, fmt="o", label="flux")
    ax.errorbar(cutout_info["central_wavelength"], cutout_info["flux"], cutout_info["flux_err"], fmt="o", label="flux")
    ax.legend()
    fig.savefig("tests/test_jax_optimizer_spherex_batch_comparison.png", dpi=300, bbox_inches='tight')
    