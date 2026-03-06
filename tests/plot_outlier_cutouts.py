import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from tractor.jax.optimizer import extract_model_data, solve_fluxes_core, render_image

from tests.test_jax_optimizer_spherex_batch import (
    FLAG_BITS,
    MASKBITS,
    BKG_MODEL,
    BKG_BOX_SIZE,
    BKG_FILTER_SIZE,
    IMG_SCALE,
    _pixel_area_sr,
    fit_background_photutils,
    fit_background_plane,
    downsample_psf_oversample2,
    _build_tractor_for_frame,
    SPHERExSersicMixture,
)
from utils import get_nearest_psf_zone_index


def _prepare_catalog():
    tab = Table.read("tests/ls_testgal.parquet")
    e1, e2 = tab["shape_e1"], tab["shape_e2"]
    e = np.hypot(e1, e2)
    ab = (1 - e) / (1 + e)
    phi = 0.5 * np.rad2deg(np.arctan2(e2, e1))
    phi = (phi + 180.0) % 180.0
    tab["shape_phi"] = phi
    tab["shape_ab"] = ab
    return tab


def _load_frame(hdul, cutout_info, idx, grid_cache):
    img_idx = 2 + idx * 6
    flg_idx = img_idx + 1
    var_idx = img_idx + 2
    bkg_idx = img_idx + 3
    psf_idx = img_idx + 4
    psf_lookup_idx = img_idx + 5

    img = hdul[img_idx].data
    flg = hdul[flg_idx].data
    var = hdul[var_idx].data
    bkg = hdul[bkg_idx].data
    psf_cube = hdul[psf_idx].data
    psf_lookup = hdul[psf_lookup_idx].data

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

    # Zone PSF selection
    gx, gy = cutout_info["x"][idx], cutout_info["y"][idx]
    zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
    zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
    psf_data = downsample_psf_oversample2(psf_cube[zidx])

    hdr = hdul[img_idx].header
    wcs = WCS(hdr)
    omega_sr = _pixel_area_sr(wcs, img.shape).astype(img.dtype, copy=False)
    img = img * omega_sr * IMG_SCALE
    bkg = bkg * omega_sr * IMG_SCALE
    var = var * (omega_sr ** 2) * (IMG_SCALE ** 2)

    frame = {
        "img": img,
        "flg": flg,
        "var": var,
        "bkg": bkg,
        "psf": psf_data,
        "hdr": hdr,
        "idx": idx,
        "omega_sr": omega_sr,
    }
    return frame


def _brightest_pixel_offset(frame, tab):
    """Return brightest pixel position and nearest catalog offset in pixels."""
    img = frame["img"]
    bkg = frame["bkg"]
    data = img - bkg

    if data.size == 0:
        return None

    data = np.array(data, copy=False)
    if not np.any(np.isfinite(data)):
        return None

    # Brightest finite pixel
    data_finite = data.copy()
    data_finite[~np.isfinite(data_finite)] = -np.inf
    y0, x0 = np.unravel_index(np.argmax(data_finite), data_finite.shape)

    wcs = WCS(frame["hdr"])
    sc = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    pxs, pys = wcs.world_to_pixel(sc)
    if pxs.size == 0:
        return (x0, y0, None, None)

    dx = pxs - x0
    dy = pys - y0
    r = np.hypot(dx, dy)
    j = int(np.argmin(r))
    return (x0, y0, float(r[j]), int(j))


def plot_cutout(
    idx,
    tractor_obj,
    out_dir,
    omega_sr=None,
    orig_shape=None,
    fixed_target_shape=None,
    fixed_max_factor=None,
    sampling_factor=None,
    vmax=None,
    cmap="viridis",
):
    images_data, batches, initial_fluxes = extract_model_data(
        tractor_obj,
        oversample_rendering=True,
        fit_background=True,
        fixed_target_shape=fixed_target_shape,
        fixed_max_factor=fixed_max_factor,
    )

    def _to_float64(obj, key=None):
        if isinstance(obj, dict):
            return {k: _to_float64(v, key=k) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            if key == "flux_idx":
                if obj.dtype != np.int32:
                    return obj.astype(np.int32)
                return obj
            if obj.dtype == np.complex64:
                return obj.astype(np.complex128)
            if obj.dtype == np.float32:
                return obj.astype(np.float64)
            return obj.astype(np.float64, copy=False)
        try:
            import jax.numpy as jnp
            if isinstance(obj, jnp.ndarray):
                if key == "flux_idx":
                    return obj.astype(jnp.int32)
                if obj.dtype == jnp.complex64:
                    return obj.astype(jnp.complex128)
                return obj.astype(jnp.float64)
        except Exception:
            pass
        return obj

    image_data = {k: v[0] for k, v in images_data.items() if k != "psf"}
    image_data["psf"] = {k: v[0] for k, v in images_data["psf"].items()}

    def _slice_batch_dict(batch):
        sliced = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced[k] = _slice_batch_dict(v)
            else:
                if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1:
                    sliced[k] = v[0]
                else:
                    sliced[k] = v
        return sliced

    single_batches = {k: _slice_batch_dict(v) for k, v in batches.items()}
    if "Background" in single_batches:
        bg_idx = single_batches["Background"]["flux_idx"]
        if not hasattr(bg_idx, "shape") or bg_idx.ndim == 0:
            single_batches["Background"]["flux_idx"] = np.array([bg_idx], dtype=np.int32)

    image_data = _to_float64(image_data)
    single_batches = _to_float64(single_batches)
    fluxes = solve_fluxes_core(
        initial_fluxes[0].astype(np.float64),
        image_data,
        single_batches,
    )
    model = render_image(
        fluxes,
        image_data,
        single_batches,
        sampling_factor=sampling_factor,
    )
    if sampling_factor is not None:
        model_native = render_image(
            fluxes,
            image_data,
            single_batches,
            sampling_factor=None,
        )
        denom = max(np.max(np.abs(model_native)), 1e-20)
        render_rel = float(np.max(np.abs(np.array(model) - np.array(model_native))) / denom)
    else:
        model_native = None
        render_rel = None

    data_full = np.array(image_data["data"])
    model_full = np.array(model)
    invvar_full = np.array(image_data["invvar"])

    # Ensure NaNs do not poison residuals/chi2: mask them out
    nan_mask = ~np.isfinite(data_full) | ~np.isfinite(invvar_full)
    if np.any(nan_mask):
        data_full = data_full.copy()
        invvar_full = invvar_full.copy()
        data_full[nan_mask] = 0.0
        invvar_full[nan_mask] = 0.0

    resid_full = data_full - model_full

    if orig_shape is not None:
        h, w = orig_shape
        sl = (slice(0, h), slice(0, w))
        data = data_full[sl]
        model = model_full[sl]
        resid = resid_full[sl]
        invvar = invvar_full[sl]
    else:
        data = data_full
        model = model_full
        resid = resid_full
        invvar = invvar_full

    chi2 = float(np.sum(resid * resid * invvar))
    n_eff = float(np.sum(invvar > 0))
    nan_invvar = int(np.sum(~np.isfinite(invvar)))
    nan_resid = int(np.sum(~np.isfinite(resid)))
    peak_data = float(np.nanmax(np.abs(data)))
    peak_model = float(np.nanmax(np.abs(model)))

    data_raw = None
    model_raw = None
    resid_raw = None
    if omega_sr is not None:
        if orig_shape is not None:
            omega_pad = omega_sr
        else:
            if omega_sr.shape != data.shape:
                pad_h = data.shape[0] - omega_sr.shape[0]
                pad_w = data.shape[1] - omega_sr.shape[1]
                omega_pad = np.pad(omega_sr, ((0, pad_h), (0, pad_w)), constant_values=0.0)
            else:
                omega_pad = omega_sr
        scale = omega_pad * IMG_SCALE
        mask = scale > 0
        data_raw = np.zeros_like(data)
        model_raw = np.zeros_like(model)
        resid_raw = np.zeros_like(resid)
        data_raw[mask] = data[mask] / scale[mask]
        model_raw[mask] = model[mask] / scale[mask]
        resid_raw[mask] = resid[mask] / scale[mask]

    # if vmax is None:
    #     vmax = max(np.max(np.abs(data)), np.max(np.abs(model)))
    # vmin = -vmax
    # from astropy.visualization import ImageNormalize, ZScaleInterval
    # norm = ImageNormalize(data, interval=ZScaleInterval())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im = axes[0].imshow(data, origin="lower", cmap=cmap)
    axes[0].set_title(f"data (img-bkg) idx={idx}")
    # get vmin, vmax from axes[0]
    vmin, vmax = im.get_clim()
    axes[1].imshow(model, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("model")

    axes[2].imshow(resid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("residual")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cutout_{idx:04d}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    
    np.save(os.path.join(out_dir, f"cutout_{idx:04d}_data.npy"), data_full)
    np.save(os.path.join(out_dir, f"cutout_{idx:04d}_model.npy"), model_full)
    np.save(os.path.join(out_dir, f"cutout_{idx:04d}_resid.npy"), resid_full)
    if orig_shape is not None:
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_data_valid.npy"), data)
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_model_valid.npy"), model)
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_resid_valid.npy"), resid)
    if model_native is not None:
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_model_native.npy"), np.array(model_native))
    if data_raw is not None:
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_data_raw.npy"), data_raw)
    if model_raw is not None:
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_model_raw.npy"), model_raw)
    if resid_raw is not None:
        np.save(os.path.join(out_dir, f"cutout_{idx:04d}_resid_raw.npy"), resid_raw)

    if render_rel is None:
        print(
            f"[cutout {idx:04d}] chi2={chi2:.3e} n_eff={n_eff:.0f} "
            f"nan_invvar={nan_invvar} nan_resid={nan_resid} "
            f"peak_data={peak_data:.3e} peak_model={peak_model:.3e}"
        )
    else:
        print(
            f"[cutout {idx:04d}] chi2={chi2:.3e} n_eff={n_eff:.0f} "
            f"render_rel_maxdiff={render_rel:.3e} "
            f"nan_invvar={nan_invvar} nan_resid={nan_resid} "
            f"peak_data={peak_data:.3e} peak_model={peak_model:.3e}"
        )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", default="tests/testphot.fits")
    parser.add_argument(
        "--cutouts",
        type=int,
        nargs="+",
        default=[127, 174, 402, 2796, 2854, 2924],
        help="Cutout indices (1-based, matching IMAGE{cutout_index})",
    )
    parser.add_argument("--out-dir", default="tests/outlier_cutout_plots")
    args = parser.parse_args()

    tab = _prepare_catalog()

    with fits.open(args.fits, memmap=True) as hdul:
        cutout_info = Table(hdul[1].data)

        grid_cache = {}
        out_paths = []
        for cutout_index in args.cutouts:
            idx = cutout_index - 1
            frame = _load_frame(hdul, cutout_info, idx, grid_cache)

            offset_info = _brightest_pixel_offset(frame, tab)
            if offset_info is None:
                print(f"[cutout {cutout_index:04d}] brightest pixel: none")
            else:
                x0, y0, r, j = offset_info
                if r is None:
                    print(
                        f"[cutout {cutout_index:04d}] brightest pixel at (x={x0}, y={y0}) "
                        f"no catalog sources"
                    )
                else:
                    print(
                        f"[cutout {cutout_index:04d}] brightest pixel at (x={x0}, y={y0}) "
                        f"nearest catalog idx={j} sep_pix={r:.2f}"
                    )

            img = frame["img"]
            psf = frame["psf"]
            max_h, max_w = img.shape
            max_psf_h, max_psf_w = psf.shape

            max_mog_K = 0
            for row in tab:
                if row["shape_r"] > 0:
                    profile_mog = SPHERExSersicMixture.getProfile(row["sersic"])
                    max_mog_K = max(max_mog_K, len(profile_mog.amp))

            tractor_obj = _build_tractor_for_frame(
                frame,
                tab,
                max_h=max_h,
                max_w=max_w,
                max_psf_h=max_psf_h,
                max_psf_w=max_psf_w,
                max_mog_K=max_mog_K,
                thaw_shape=False,
                thaw_positions=False,
                maskbits=MASKBITS,
            )

            fixed_max_factor = 5.0
            fft_pad_h_lr = int(np.ceil(max_psf_h / fixed_max_factor))
            fft_pad_w_lr = int(np.ceil(max_psf_w / fixed_max_factor))
            padded_h = max_h + fft_pad_h_lr
            padded_w = max_w + fft_pad_w_lr
            fixed_target_shape = (
                int(round(padded_h * fixed_max_factor)),
                int(round(padded_w * fixed_max_factor)),
            )

            out_path = plot_cutout(
                cutout_index,
                tractor_obj,
                args.out_dir,
                omega_sr=frame["omega_sr"],
                orig_shape=frame["img"].shape,
                fixed_target_shape=fixed_target_shape,
                fixed_max_factor=fixed_max_factor,
                sampling_factor=fixed_max_factor,
            )
            out_paths.append(out_path)

    print("Saved plots:")
    for p in out_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
