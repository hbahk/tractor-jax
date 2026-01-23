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
from tractor.jax.optimizer import JaxOptimizer, extract_model_data, render_image
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

def test_jax_optimizer_accuracy():
    print(jax.devices())
    print(jax.default_backend())
    
    cutout_size = 6.15 * 15 * u.arcsec
    gra, gdec = 258.2084186 * u.deg, 64.0529535 * u.deg
    gco = SkyCoord(ra=gra, dec=gdec)
    output_filename = "tests/testphot.fits"
    
    tab = Table.read("tests/ls_testgal.parquet")
    tco = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")

    
    # open the output file
    start_time = time.time()
    hdul = fits.open(output_filename)
    end_time = time.time()
    print(f"Time to open the output file: {end_time - start_time} seconds")

    start_time = time.time()
    cutout_info = Table(hdul[1].data)
    end_time = time.time()
    print(f"Time to read the cutout info: {end_time - start_time} seconds")

    start_time = time.time()
    cutout_index = 1446

    img = hdul[f"IMAGE{cutout_index}"].data
    hdr = hdul[f"IMAGE{cutout_index}"].header
    wcs = WCS(hdr)

    flg = hdul[f"FLAGS{cutout_index}"].data
    var = hdul[f"VARIANCE{cutout_index}"].data
    bkg = hdul[f"ZODI{cutout_index}"].data

    invvar = 1 / var
    mask = flg & MASKBITS != 0
    invvar[mask] = 0

    psf_cube = hdul[f"PSF{cutout_index}"].data
    psf_lookup = hdul[f"PSF_ZONE_LOOKUP{cutout_index}"].data
    end_time = time.time()
    print(f"Time to get the frames: {end_time - start_time} seconds")

    start_time = time.time()
    gx, gy = cutout_info["x"][cutout_index-1], cutout_info["y"][cutout_index-1]
    zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
    zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
    psf = psf_cube[zidx]
    end_time = time.time()
    print(f"Time to get the PSF of the zone: {end_time - start_time} seconds")

    start_time = time.time()
    psf_tractor = PixelizedPSF(psf, sampling=0.1)
    end_time = time.time()
    print(f"Time to create the PSF: {end_time - start_time} seconds")

    # get the pixel coordinates of the target
    start_time = time.time()
    tx, ty = wcs.world_to_pixel(tco)
    tab["x"], tab["y"] = tx, ty

    tinside = (tx > 0) & (tx < img.shape[1]) & (ty > 0) & (ty < img.shape[0])
    end_time = time.time()
    print(f"Time to get the pixel coordinates and inside mask: {end_time - start_time} seconds")

    start_time = time.time()
    e1, e2 = tab["shape_e1"], tab["shape_e2"]
    e = np.hypot(e1, e2)
    ab = (1 - e) / (1 + e)  # axis ratio = b/a
    # phi = -np.rad2deg(np.arctan2(e2, e1) / 2)
    phi = 0.5 * np.rad2deg(np.arctan2(e2, e1))
    phi = (phi + 180.0) % 180.0


    tab["shape_phi"] = phi
    tab["shape_ab"] = ab

    end_time = time.time()
    print(f"Time to get the shape parameters: {end_time - start_time} seconds")

    start_time = time.time()
    stab = tab[tinside]
    sc = SkyCoord(ra=stab["ra"], dec=stab["dec"], unit="deg")
    sep = sc.separation(gco)
    main_idx = np.argmin(sep)
    target = stab[main_idx]
    end_time = time.time()
    print(f"Time to get the sources inside the image: {end_time - start_time} seconds")

    start_time = time.time()
    tractor_source_list = []
    for row in stab:
        _flux = Flux(np.random.uniform(high=1))
        if row["shape_r"] == 0:
            # _src = PointSource(PixPos(row["x"], row["y"]), Flux(row["flux_z"]))
            _src = PointSource(PixPos(row["x"], row["y"]), _flux)
        else:
            # cos = wcs.celestial.wcs.get_pc()[0, 0]
            # sin = wcs.celestial.wcs.get_pc()[1, 0]
            # angle = np.arctan2(sin, cos) * 180 / np.pi
            
            phi_img = sky_pa_to_pixel_pa(wcs, row["ra"], row["dec"], row["shape_phi"], d_arcsec=1.0, y_down=False)
            
            _src = SPHERExSersicGalaxy(
                PixPos(row["x"], row["y"]),
                # Flux(row["flux_z"]),
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
        
    end_time = time.time()
    print(f"Time to construct the source list: {end_time - start_time} seconds")

    start_time = time.time()
    tim = Image(
        data=img - bkg,
        inverr=np.sqrt(invvar),
        psf=psf_tractor,
        wcs=NullWCS(pixscale=6.15),
        photocal=LinearPhotoCal(1.0),
        sky=ConstantSky(0.0),
    )

    tim.freezeAllRecursive()
    # tim.thawParam("sky")
    tim.thawPathsTo("sky")

    end_time = time.time()
    print(f"Time to construct the Tractor image: {end_time - start_time} seconds")

    start_time = time.time()
    trac_spherex = tractor.Tractor([tim], tractor_source_list)
    end_time = time.time()
    print(f"Time to construct the Tractor object: {end_time - start_time} seconds")

    # Perturb fluxes
    for i, src in enumerate(trac_spherex.catalog):
        src.brightness.setParams([np.random.uniform(high=1)])

    # print("Initial Fluxes:", [src.brightness.getParams()[0] for src in trac_spherex.catalog])

    # Use JaxOptimizer
    trac_spherex.optimizer = JaxOptimizer()
    # We must inform JaxOptimizer to use oversampling.
    # The optimize method takes kwargs.
    # But currently JaxOptimizer hardcodes oversample_rendering=True?
    # Yes, I wrote it to pass oversample_rendering=True.

    # Run optimization
    dchisq = 1e-10
    for step in range(10):
        # We need variance=True to match user loop signature, though we don't use it here
        flux = trac_spherex.catalog[main_idx].brightness.getParams()[0] * PIX_SR * 1.0e9
        print(f"Before Step {step}: Flux={flux:.6f}")
        dlnp, X, alpha, var = trac_spherex.optimize(variance=True, shared_params=False)
        # print(f"Step {step}: dlnp={dlnp}")
        if dlnp < dchisq:
            break

    flux = trac_spherex.catalog[main_idx].brightness.getParams()[0] * PIX_SR * 1.0e9
    print(f"Final Flux: {flux:.6f}")
    
    mod = trac_spherex.getModelImage(0)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img-bkg, origin='lower', vmin=0, vmax=1)
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(mod, origin='lower', vmin=0, vmax=1)
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img-bkg-mod, origin='lower', vmin=0, vmax=1)
    
    fig.savefig("tests/test_jax_optimizer_spherex.png", dpi=300, bbox_inches='tight')
    

def test_jax_optimizer_multiple(idx_list):
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

        tinside = (tx > -0.5) & (tx < img.shape[1]+0.5) & (ty > -0.5) & (ty < img.shape[0]+0.5)

        gra, gdec = 258.2084186 * u.deg, 64.0529535 * u.deg
        gco = SkyCoord(ra=gra, dec=gdec)

        stab = tab[tinside]
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
            
        trac_spherex = tractor.Tractor([tim], tractor_source_list)
        
        trac_spherex.optimizer = JaxOptimizer()
        dchisq = 1e-10
        for step in range(10):
            flux = trac_spherex.catalog[main_idx].brightness.getParams()[0] * PIX_SR * 1.0e9
            print(f"Before Step {step}: Flux={flux:.6f}")
            dlnp, X, alpha, var = trac_spherex.optimize(variance=True, shared_params=False)
            if dlnp < dchisq:
                break

        flux = trac_spherex.catalog[main_idx].brightness.getParams()[0] * PIX_SR * 1.0e9
        ferr = np.sqrt(var)[main_idx] * PIX_SR * 1.0e9
        print(f"Final Flux: {flux:.6f}, Flux Error: {ferr:.6f}")
        cutout_info["flux"][i] = flux
        cutout_info["flux_err"][i] = ferr
        
    return cutout_info[idx_list]

if __name__ == "__main__":
    # test_jax_optimizer_accuracy()
    
    # test_index = np.arange(0, 3000, 60)
    test_index = np.arange(0, 3000, 300)
    cutout_info = test_jax_optimizer_multiple(test_index)
    cutout_info.write("tests/test_jax_optimizer_spherex_radec.parquet", overwrite=True)
    
    cpu_result = Table.read("/data1/hbahk/spherex-cluster/codes/realworld/specphot_results_testgal_a2255.parquet")

    wave = cpu_result["central_wavelength"]
    flux = cpu_result["flux"]
    ferr = cpu_result["flux_err"]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(wave, flux, ferr, fmt="o", label="flux")
    ax.errorbar(cutout_info["central_wavelength"], cutout_info["flux"], cutout_info["flux_err"], fmt="o", label="flux")
    ax.legend()
    fig.savefig("tests/test_jax_optimizer_spherex_comparison_radec.png", dpi=300, bbox_inches='tight')
    