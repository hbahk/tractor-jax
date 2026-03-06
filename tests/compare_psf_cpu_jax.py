import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from utils import get_nearest_psf_zone_index

import jax.numpy as jnp
import jax.numpy.fft as jfft
from tractor.jax.rendering import (
    render_point_source_fft,
    render_pixelized_psf,
    rebin_downsample_int_flux,
)


def compute_centroid(img):
    yy, xx = np.indices(img.shape)
    total = img.sum()
    if total == 0:
        return (np.nan, np.nan)
    return (np.sum(yy * img) / total, np.sum(xx * img) / total)


def compute_offset(psf, mode="none", core_radius=5):
    h, w = psf.shape
    cy, cx = h // 2, w // 2

    if mode == "none":
        return (0.0, 0.0)

    if mode == "peak":
        my, mx = np.unravel_index(np.argmax(psf), psf.shape)
        return (my - cy, mx - cx)

    if mode == "centroid":
        cty, ctx = compute_centroid(psf)
        return (cty - cy, ctx - cx)

    if mode == "core":
        yy, xx = np.indices(psf.shape)
        rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = rr2 <= core_radius ** 2
        core = psf * mask
        cty, ctx = compute_centroid(core)
        return (cty - cy, ctx - cx)

    raise ValueError(f"Unknown mode: {mode}")


def summarize(label, img):
    h, w = img.shape
    cy, cx = h // 2, w // 2
    max_pos = np.unravel_index(np.argmax(img), img.shape)
    centroid = compute_centroid(img)
    return {
        "label": label,
        "shape": (h, w),
        "center_pix": (cy, cx),
        "max_pos": max_pos,
        "max_offset": (max_pos[0] - cy, max_pos[1] - cx),
        "centroid": centroid,
        "centroid_offset": (centroid[0] - cy, centroid[1] - cx),
        "sum": float(img.sum()),
    }


def format_stats(stats):
    h, w = stats["shape"]
    cy, cx = stats["center_pix"]
    my, mx = stats["max_pos"]
    cty, ctx = stats["centroid"]
    dy_m, dx_m = stats["max_offset"]
    dy_c, dx_c = stats["centroid_offset"]
    return (
        f"{stats['label']} shape={h}x{w} center=({cy},{cx}) "
        f"max=({my},{mx}) dmax=({dy_m:+.3f},{dx_m:+.3f}) "
        f"centroid=({cty:.3f},{ctx:.3f}) dcentroid=({dy_c:+.3f},{dx_c:+.3f}) "
        f"sum={stats['sum']:.6g}"
    )


def _get_hires_psf_gsobj(psf, pixscale_psf):
    import galsim
    return galsim.InterpolatedImage(
        galsim.Image(psf), scale=pixscale_psf, flux=1.0, x_interpolant="linear"
    )


def _get_psf_patch(psf_gsobj, px, py, nx, ny, pixscale):
    import galsim
    ix = round(float(px))
    iy = round(float(py))
    imgcentx = (nx - 1) / 2
    imgcenty = (ny - 1) / 2
    hx = imgcentx % 1
    hy = imgcenty % 1
    dx = px - ix + hx
    dy = py - iy + hy
    psf_image = psf_gsobj.drawImage(
        method="no_pixel", scale=pixscale, nx=nx, ny=ny, offset=galsim.PositionD(dx, dy)
    ).array
    return psf_image


def render_cpu_psf_patch(psf, px, py, oversample=10, pixscale=6.15, debug=False):
    """
    Return downsampled PSF patch and its placement origin (ix0, iy0) in image coords.
    """
    outimg, ix0, iy0, _meta, _hires = render_cpu_psf_patch_meta(
        psf,
        px,
        py,
        oversample=oversample,
        pixscale=pixscale,
        debug=debug,
        return_hires=False,
    )
    return outimg, ix0, iy0


def render_cpu_psf_patch_meta(
    psf, px, py, oversample=10, pixscale=6.15, debug=False, return_hires=False
):
    """
    Return downsampled PSF patch, origin, and CPU meta used for hires generation.
    """
    from skimage.transform import downscale_local_mean
    pixscale_psf = pixscale / oversample
    psf_gsobj = _get_hires_psf_gsobj(psf, pixscale_psf)

    size = psf_gsobj.getGoodImageSize(pixscale / oversample)
    nx = ny = size

    # oversampled pixel coordinates
    pxov = (px + 0.5) * oversample - 0.5
    pyov = (py + 0.5) * oversample - 0.5

    hires_psf = _get_psf_patch(
        psf_gsobj, pxov, pyov, nx, ny, pixscale / oversample
    )

    # Align to oversample grid (same logic as SPHERExTractorPSF)
    ipxov = round(float(pxov))
    ipyov = round(float(pyov))
    ix0_hires = ipxov - nx // 2
    iy0_hires = ipyov - ny // 2

    npad_left = ix0_hires % oversample
    hires_psf = np.c_[np.zeros((hires_psf.shape[0], npad_left)), hires_psf]
    npad_bottom = iy0_hires % oversample
    hires_psf = np.r_[np.zeros((npad_bottom, hires_psf.shape[1])), hires_psf]

    nover = hires_psf.shape[1] % oversample
    if nover > 0:
        npad_right = oversample - nover
        hires_psf = np.c_[hires_psf, np.zeros((hires_psf.shape[0], npad_right))]

    nover = hires_psf.shape[0] % oversample
    if nover > 0:
        npad_top = oversample - nover
        hires_psf = np.r_[np.zeros((npad_top, hires_psf.shape[1])), hires_psf]

    outimg = downscale_local_mean(hires_psf, (oversample, oversample))
    outimg *= oversample ** 2

    # Placement origin in image coords
    ix0 = (ix0_hires - npad_left) // oversample
    iy0 = (iy0_hires - npad_bottom) // oversample
    ph, pw = outimg.shape

    if debug:
        print(
            f"    cpu_hires: nx=ny={nx} ph=pw={ph} "
            f"pxov={pxov:.3f} pyov={pyov:.3f} "
            f"ipxov={ipxov} ipyov={ipyov} "
            f"ix0_hires={ix0_hires} iy0_hires={iy0_hires} "
            f"npad_left={npad_left} npad_bottom={npad_bottom} "
            f"ix0={ix0} iy0={iy0} outshape={outimg.shape}"
        )

    meta = {
        "nx": int(nx),
        "ny": int(ny),
        "pxov": float(pxov),
        "pyov": float(pyov),
        "ipxov": int(ipxov),
        "ipyov": int(ipyov),
        "ix0_hires": int(ix0_hires),
        "iy0_hires": int(iy0_hires),
        "npad_left": int(npad_left),
        "npad_bottom": int(npad_bottom),
    }
    if return_hires:
        return outimg.astype(np.float32), int(ix0), int(iy0), meta, hires_psf.astype(np.float32)
    return outimg.astype(np.float32), int(ix0), int(iy0), meta, None


def render_cpu_psf(psf, target_shape, px, py, oversample=10, pixscale=6.15):
    outimg, ix0, iy0 = render_cpu_psf_patch(psf, px, py, oversample=oversample, pixscale=pixscale)

    img = np.zeros(target_shape, dtype=np.float32)
    h, w = target_shape
    ph, pw = outimg.shape
    y0, x0 = iy0, ix0
    y1, x1 = y0 + ph, x0 + pw

    yy0 = max(y0, 0)
    xx0 = max(x0, 0)
    yy1 = min(y1, h)
    xx1 = min(x1, w)

    if yy0 >= yy1 or xx0 >= xx1:
        return img

    src_y0 = yy0 - y0
    src_x0 = xx0 - x0
    src_y1 = src_y0 + (yy1 - yy0)
    src_x1 = src_x0 + (xx1 - xx0)

    img[yy0:yy1, xx0:xx1] = outimg[src_y0:src_y1, src_x0:src_x1]
    return img


def render_jax_psf(psf, target_shape, px, py):
    h, w = target_shape
    ph, pw = psf.shape
    cy, cx = h // 2, w // 2
    y0 = cy - ph // 2
    x0 = cx - pw // 2

    pad_img = jnp.zeros((h, w))
    pad_img = pad_img.at[y0 : y0 + ph, x0 : x0 + pw].set(jnp.array(psf))
    pad_img = jnp.fft.ifftshift(pad_img)
    psf_fft = jfft.rfft2(pad_img)

    pos = jnp.array([px, py])
    img = render_point_source_fft(1.0, pos, psf_fft, (h, w))
    return np.array(img)


def _bilinear_sample(img, yy, xx):
    H, W = img.shape
    y0 = jnp.floor(yy).astype(jnp.int32)
    x0 = jnp.floor(xx).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = yy - y0
    wx = xx - x0

    def _gather(y, x):
        y_clip = jnp.clip(y, 0, H - 1)
        x_clip = jnp.clip(x, 0, W - 1)
        val = img[y_clip, x_clip]
        mask = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        return jnp.where(mask, val, 0.0)

    v00 = _gather(y0, x0)
    v01 = _gather(y0, x1)
    v10 = _gather(y1, x0)
    v11 = _gather(y1, x1)

    return (
        (1.0 - wy) * (1.0 - wx) * v00
        + (1.0 - wy) * wx * v01
        + wy * (1.0 - wx) * v10
        + wy * wx * v11
    )


def render_jax_hires_from_psf(psf, nx, ny, dx, dy, src_shift=(0.0, 0.0), sign=1.0):
    """
    JAX linear interpolation to sample PSF on hires grid with subpixel offset.
    """
    psf_img = jnp.array(psf)
    H, W = psf_img.shape

    cy_out = (ny - 1) / 2.0
    cx_out = (nx - 1) / 2.0
    cy_in = (H - 1) / 2.0
    cx_in = (W - 1) / 2.0

    yy = jnp.arange(ny, dtype=jnp.float32)
    xx = jnp.arange(nx, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(yy, xx, indexing="ij")

    # Shift output grid by (dx, dy) in hires pixel units.
    shift_y, shift_x = src_shift
    src_y = (yy - cy_out) + cy_in - sign * dy + shift_y
    src_x = (xx - cx_out) + cx_in - sign * dx + shift_x

    return _bilinear_sample(psf_img, src_y, src_x)


def render_jax_psf_cpu_hires(
    psf,
    target_shape,
    px,
    py,
    oversample=10,
    cpu_meta=None,
    cpu_hires=None,
    jax_hires_shift=(0.0, 0.0),
    jax_hires_sign=1.0,
):
    """
    Generate hires grid with CPU nx,ny and CPU-style offsets, then downsample like CPU.
    If cpu_hires is provided, it is used directly for the downsample path.
    """
    if cpu_meta is None:
        raise ValueError("cpu_meta required for cpu_hires/jax_hires modes")

    nx = int(cpu_meta["nx"])
    ny = int(cpu_meta["ny"])

    # Oversampled coordinates (CPU)
    pxov = float(cpu_meta.get("pxov", (px + 0.5) * oversample - 0.5))
    pyov = float(cpu_meta.get("pyov", (py + 0.5) * oversample - 0.5))
    ipxov = int(cpu_meta.get("ipxov", round(pxov)))
    ipyov = int(cpu_meta.get("ipyov", round(pyov)))

    imgcentx = (nx - 1) / 2.0
    imgcenty = (ny - 1) / 2.0
    hx = imgcentx % 1
    hy = imgcenty % 1
    dx = pxov - ipxov + hx
    dy = pyov - ipyov + hy

    if cpu_hires is None:
        hires = render_jax_hires_from_psf(
            psf, nx, ny, dx, dy, src_shift=jax_hires_shift, sign=jax_hires_sign
        )
    else:
        hires = jnp.array(cpu_hires)

    ix0_hires = int(cpu_meta.get("ix0_hires", ipxov - nx // 2))
    iy0_hires = int(cpu_meta.get("iy0_hires", ipyov - ny // 2))
    npad_left = int(cpu_meta.get("npad_left", ix0_hires % oversample))
    npad_bottom = int(cpu_meta.get("npad_bottom", iy0_hires % oversample))

    if npad_left:
        hires = jnp.pad(hires, ((0, 0), (npad_left, 0)))
    if npad_bottom:
        hires = jnp.pad(hires, ((npad_bottom, 0), (0, 0)))

    pad_h = (-hires.shape[0]) % oversample
    pad_w = (-hires.shape[1]) % oversample
    if pad_h or pad_w:
        hires = jnp.pad(hires, ((0, pad_h), (0, pad_w)))

    ds = rebin_downsample_int_flux(hires, oversample, oversample)

    ix0 = (ix0_hires - npad_left) // oversample
    iy0 = (iy0_hires - npad_bottom) // oversample

    h, w = target_shape
    dh, dw = ds.shape
    y0, x0 = iy0, ix0

    yy0 = max(y0, 0)
    xx0 = max(x0, 0)
    yy1 = min(y0 + dh, h)
    xx1 = min(x0 + dw, w)

    if yy0 >= yy1 or xx0 >= xx1:
        pad_img = jnp.zeros((h, w))
    else:
        src_y0 = yy0 - y0
        src_x0 = xx0 - x0
        src_y1 = src_y0 + (yy1 - yy0)
        src_x1 = src_x0 + (xx1 - xx0)
        pad_img = jnp.zeros((h, w))
        pad_img = pad_img.at[yy0:yy1, xx0:xx1].set(ds[src_y0:src_y1, src_x0:src_x1])

    pad_img = jnp.fft.ifftshift(pad_img)
    psf_fft = jfft.rfft2(pad_img)

    pos = jnp.array([px, py])
    img = render_point_source_fft(1.0, pos, psf_fft, (h, w))
    return np.array(img)


def render_jax_psf_psfshift(psf, target_shape, px, py, oversample=10, origin_override=None):
    """
    Shift oversampled PSF by subpixel offsets, then downsample before FFT.
    Mimics CPU oversample -> shift -> downsample -> placement pipeline.
    """
    h, w = target_shape
    ph, pw = psf.shape

    # CPU-style oversampled coordinates
    pxov = (px + 0.5) * oversample - 0.5
    pyov = (py + 0.5) * oversample - 0.5

    # Subpixel offset within oversampled grid
    dx = pxov - np.round(pxov)
    dy = pyov - np.round(pyov)

    shifted = render_pixelized_psf(jnp.array(psf), dx, dy)

    # CPU-style patch origin in oversampled grid
    ipxov = int(np.round(pxov))
    ipyov = int(np.round(pyov))
    ix0_hires = ipxov - ph // 2
    iy0_hires = ipyov - pw // 2

    # Pad to align on oversample grid (same as CPU)
    npad_left = ix0_hires % oversample
    npad_bottom = iy0_hires % oversample
    if npad_left:
        shifted = jnp.pad(shifted, ((0, 0), (npad_left, 0)))
    if npad_bottom:
        shifted = jnp.pad(shifted, ((npad_bottom, 0), (0, 0)))

    # Pad to make divisible by oversample
    pad_h = (-shifted.shape[0]) % oversample
    pad_w = (-shifted.shape[1]) % oversample
    if pad_h or pad_w:
        shifted = jnp.pad(shifted, ((0, pad_h), (0, pad_w)))

    # Downsample (sum) by integer factor
    ds = rebin_downsample_int_flux(shifted, oversample, oversample)

    # CPU-style placement in target image
    ix0 = (ix0_hires - npad_left) // oversample
    iy0 = (iy0_hires - npad_bottom) // oversample
    dh, dw = ds.shape
    if origin_override is not None:
        x0, y0 = origin_override
    else:
        y0 = iy0
        x0 = ix0

    # Clip to target bounds
    yy0 = max(y0, 0)
    xx0 = max(x0, 0)
    yy1 = min(y0 + dh, h)
    xx1 = min(x0 + dw, w)

    if yy0 >= yy1 or xx0 >= xx1:
        pad_img = jnp.zeros((h, w))
    else:
        src_y0 = yy0 - y0
        src_x0 = xx0 - x0
        src_y1 = src_y0 + (yy1 - yy0)
        src_x1 = src_x0 + (xx1 - xx0)
        pad_img = jnp.zeros((h, w))
        pad_img = pad_img.at[yy0:yy1, xx0:xx1].set(ds[src_y0:src_y1, src_x0:src_x1])

    pad_img = jnp.fft.ifftshift(pad_img)
    psf_fft = jfft.rfft2(pad_img)

    pos = jnp.array([px, py])
    img = render_point_source_fft(1.0, pos, psf_fft, (h, w))
    return np.array(img)


def render_jax_from_cpu_patch(cpu_patch, cpu_ix0, cpu_iy0, target_shape, px, py):
    """
    Use CPU downsampled patch directly as FFT input in JAX.
    """
    h, w = target_shape
    ph, pw = cpu_patch.shape

    y0, x0 = cpu_iy0, cpu_ix0
    y1, x1 = y0 + ph, x0 + pw

    yy0 = max(y0, 0)
    xx0 = max(x0, 0)
    yy1 = min(y1, h)
    xx1 = min(x1, w)

    if yy0 >= yy1 or xx0 >= xx1:
        pad_img = jnp.zeros((h, w))
    else:
        src_y0 = yy0 - y0
        src_x0 = xx0 - x0
        src_y1 = src_y0 + (yy1 - yy0)
        src_x1 = src_x0 + (xx1 - xx0)
        pad_img = jnp.zeros((h, w))
        pad_img = pad_img.at[yy0:yy1, xx0:xx1].set(
            jnp.array(cpu_patch[src_y0:src_y1, src_x0:src_x1])
        )

    pad_img = jnp.fft.ifftshift(pad_img)
    psf_fft = jfft.rfft2(pad_img)

    pos = jnp.array([px, py])
    img = render_point_source_fft(1.0, pos, psf_fft, (h, w))
    return np.array(img)


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
    parser.add_argument("--dx", type=float, default=0.0, help="Subpixel x offset")
    parser.add_argument("--dy", type=float, default=0.0, help="Subpixel y offset")
    parser.add_argument("--save-dir", default=None, help="Directory to save npy outputs")
    parser.add_argument(
        "--mode",
        choices=["cpu", "jax", "both"],
        default="both",
        help="Which renderer to run",
    )
    parser.add_argument(
        "--offset-mode",
        choices=[
            "none",
            "centroid",
            "peak",
            "core",
            "psfshift",
            "psfshift_os",
            "cpu_patch",
            "cpu_origin",
            "cpu_hires",
            "jax_hires",
        ],
        default="none",
        help="Offset mode applied to JAX positions",
    )
    parser.add_argument(
        "--core-radius",
        type=float,
        default=5.0,
        help="Core radius (pixels) for core-centroid offset",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=10,
        help="Oversample factor for psfshift/psfshift_os",
    )
    parser.add_argument(
        "--cpu-dir",
        default=None,
        help="Directory with saved cpu_{cutout}.npy for diff",
    )
    parser.add_argument(
        "--save-cpu-patch",
        action="store_true",
        help="Save CPU downsampled patch + origin for later reuse",
    )
    parser.add_argument(
        "--save-cpu-hires",
        action="store_true",
        help="Save CPU hires PSF and meta for JAX hires tests",
    )
    parser.add_argument(
        "--debug-cpu",
        action="store_true",
        help="Print CPU hires grid/origin diagnostics",
    )
    parser.add_argument(
        "--cpu-patch-dir",
        default=None,
        help="Directory with saved cpu_patch_{cutout}.npz",
    )
    parser.add_argument(
        "--jax-hires-shift",
        type=float,
        nargs=2,
        default=(-1.0, 0.5),
        metavar=("DY", "DX"),
        help="Extra (dy, dx) shift applied to JAX hires sampling",
    )
    parser.add_argument(
        "--jax-hires-sign",
        choices=["minus", "plus"],
        default="minus",
        help="Sign convention for applying (dx, dy) in JAX hires sampling",
    )
    parser.add_argument(
        "--jax-hires-sweep",
        action="store_true",
        help="Sweep jax-hires shift grid and report best L1/L2",
    )
    parser.add_argument(
        "--jax-hires-sweep-step",
        type=float,
        default=0.5,
        help="Grid step for jax-hires sweep (in hires pixels)",
    )
    parser.add_argument(
        "--jax-hires-sweep-range",
        type=float,
        default=1.0,
        help="Half-range for jax-hires sweep (in hires pixels)",
    )
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    with fits.open(args.fits, memmap=True) as hdul:
        cutout_info = Table(hdul[1].data)

        for cutout_index in args.cutouts:
            img_hdu = f"IMAGE{cutout_index}"
            psf_hdu = f"PSF{cutout_index}"
            lookup_hdu = f"PSF_ZONE_LOOKUP{cutout_index}"

            if img_hdu not in hdul or psf_hdu not in hdul or lookup_hdu not in hdul:
                print(f"[{cutout_index}] missing HDU(s), skipping")
                continue

            psf_cube = hdul[psf_hdu].data
            psf_lookup = hdul[lookup_hdu].data

            row = cutout_info[cutout_index - 1]
            gx, gy = row["x"], row["y"]
            zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
            zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
            psf = psf_cube[zidx]

            target_shape = psf.shape
            cy, cx = target_shape[0] // 2, target_shape[1] // 2
            px = cx + args.dx
            py = cy + args.dy

            print(f"\n[{cutout_index}] zone_id={zoneid} zidx={zidx} pos=({px:.3f},{py:.3f})")
            cpu_img = None
            jax_img = None
            delta = (0.0, 0.0)

            if args.mode in ["cpu", "both"]:
                cpu_img = render_cpu_psf(psf, target_shape, px, py)
                cpu_stats = summarize("cpu", cpu_img)
                print("  " + format_stats(cpu_stats))
                if (args.save_cpu_patch or args.save_cpu_hires) and args.save_dir:
                    patch, ix0, iy0, meta, hires = render_cpu_psf_patch_meta(
                        psf,
                        px,
                        py,
                        debug=args.debug_cpu,
                        return_hires=args.save_cpu_hires,
                    )
                    save_kwargs = {
                        "patch": patch,
                        "ix0": ix0,
                        "iy0": iy0,
                        "nx": meta["nx"],
                        "ny": meta["ny"],
                        "pxov": meta["pxov"],
                        "pyov": meta["pyov"],
                        "ipxov": meta["ipxov"],
                        "ipyov": meta["ipyov"],
                        "ix0_hires": meta["ix0_hires"],
                        "iy0_hires": meta["iy0_hires"],
                        "npad_left": meta["npad_left"],
                        "npad_bottom": meta["npad_bottom"],
                    }
                    if hires is not None:
                        save_kwargs["hires"] = hires
                    np.savez(
                        os.path.join(args.save_dir, f"cpu_patch_{cutout_index}.npz"),
                        **save_kwargs,
                    )

            if args.mode in ["jax", "both"]:
                if args.offset_mode == "cpu_patch":
                    if not args.cpu_patch_dir:
                        raise ValueError("--cpu-patch-dir required for cpu_patch mode")
                    patch_path = os.path.join(args.cpu_patch_dir, f"cpu_patch_{cutout_index}.npz")
                    if not os.path.exists(patch_path):
                        raise FileNotFoundError(patch_path)
                    data = np.load(patch_path)
                    jax_img = render_jax_from_cpu_patch(
                        data["patch"], int(data["ix0"]), int(data["iy0"]), target_shape, px, py
                    )
                elif args.offset_mode == "cpu_origin":
                    if not args.cpu_patch_dir:
                        raise ValueError("--cpu-patch-dir required for cpu_origin mode")
                    patch_path = os.path.join(args.cpu_patch_dir, f"cpu_patch_{cutout_index}.npz")
                    if not os.path.exists(patch_path):
                        raise FileNotFoundError(patch_path)
                    data = np.load(patch_path)
                    origin = (int(data["ix0"]), int(data["iy0"]))
                    jax_img = render_jax_psf_psfshift(
                        psf,
                        target_shape,
                        px,
                        py,
                        oversample=args.oversample,
                        origin_override=origin,
                    )
                elif args.offset_mode in ["cpu_hires", "jax_hires"]:
                    if not args.cpu_patch_dir:
                        raise ValueError("--cpu-patch-dir required for cpu_hires/jax_hires mode")
                    patch_path = os.path.join(args.cpu_patch_dir, f"cpu_patch_{cutout_index}.npz")
                    if not os.path.exists(patch_path):
                        raise FileNotFoundError(patch_path)
                    data = np.load(patch_path)
                    meta = {
                        "nx": int(data["nx"]),
                        "ny": int(data["ny"]),
                        "pxov": float(data["pxov"]),
                        "pyov": float(data["pyov"]),
                        "ipxov": int(data["ipxov"]),
                        "ipyov": int(data["ipyov"]),
                        "ix0_hires": int(data["ix0_hires"]),
                        "iy0_hires": int(data["iy0_hires"]),
                        "npad_left": int(data["npad_left"]),
                        "npad_bottom": int(data["npad_bottom"]),
                    }
                    if args.offset_mode == "cpu_hires" and "hires" not in data:
                        raise ValueError("--save-cpu-hires required to use cpu_hires mode")
                    cpu_hires = data["hires"] if args.offset_mode == "cpu_hires" else None
                    if args.offset_mode == "jax_hires" and args.jax_hires_sweep:
                        dy_vals = np.arange(
                            -args.jax_hires_sweep_range,
                            args.jax_hires_sweep_range + 1e-6,
                            args.jax_hires_sweep_step,
                        )
                        dx_vals = np.arange(
                            -args.jax_hires_sweep_range,
                            args.jax_hires_sweep_range + 1e-6,
                            args.jax_hires_sweep_step,
                        )
                        best = None
                        for dy in dy_vals:
                            for dx in dx_vals:
                                cand = render_jax_psf_cpu_hires(
                                    psf,
                                    target_shape,
                                    px,
                                    py,
                                    oversample=args.oversample,
                                    cpu_meta=meta,
                                    cpu_hires=cpu_hires,
                                    jax_hires_shift=(float(dy), float(dx)),
                                    jax_hires_sign=(
                                        1.0 if args.jax_hires_sign == "minus" else -1.0
                                    ),
                                )
                                if cpu_img is None and args.cpu_dir:
                                    cpu_path = os.path.join(
                                        args.cpu_dir, f"cpu_{cutout_index}.npy"
                                    )
                                    if os.path.exists(cpu_path):
                                        cpu_img = np.load(cpu_path)
                                if cpu_img is None:
                                    raise ValueError(
                                        "--cpu-dir with cpu_{cutout}.npy required for sweep"
                                    )
                                diff = cand - cpu_img
                                diff_l2 = np.sqrt(np.mean(diff**2))
                                diff_l1 = np.mean(np.abs(diff))
                                score = (diff_l1, diff_l2)
                                if best is None or score < best["score"]:
                                    best = {
                                        "score": score,
                                        "shift": (float(dy), float(dx)),
                                        "img": cand,
                                    }
                        jax_img = best["img"]
                        print(
                            f"  jax_hires sweep best shift=({best['shift'][0]:+.3f},{best['shift'][1]:+.3f}) "
                            f"L1={best['score'][0]:.6g} L2={best['score'][1]:.6g}"
                        )
                    else:
                        jax_img = render_jax_psf_cpu_hires(
                            psf,
                            target_shape,
                            px,
                            py,
                            oversample=args.oversample,
                            cpu_meta=meta,
                            cpu_hires=cpu_hires,
                            jax_hires_shift=tuple(args.jax_hires_shift),
                            jax_hires_sign=(1.0 if args.jax_hires_sign == "minus" else -1.0),
                        )
                elif args.offset_mode in ["psfshift", "psfshift_os"]:
                    jax_img = render_jax_psf_psfshift(
                        psf, target_shape, px, py, oversample=args.oversample
                    )
                else:
                    delta = compute_offset(psf, mode=args.offset_mode, core_radius=args.core_radius)
                    py_jax = py + delta[0]
                    px_jax = px + delta[1]
                    jax_img = render_jax_psf(psf, target_shape, px_jax, py_jax)
                jax_stats = summarize("jax", jax_img)
                print("  " + format_stats(jax_stats))
                if args.offset_mode in ["centroid", "peak", "core"]:
                    print(f"  jax offset mode={args.offset_mode} delta=({delta[0]:+.3f},{delta[1]:+.3f})")
                if args.offset_mode == "psfshift":
                    print("  jax offset mode=psfshift (subpixel PSF shift before FFT)")
                if args.offset_mode == "psfshift_os":
                    print("  jax offset mode=psfshift_os (oversample shift+downsample before FFT)")
                if args.offset_mode == "cpu_patch":
                    print("  jax offset mode=cpu_patch (CPU patch -> JAX FFT)")
                if args.offset_mode == "cpu_origin":
                    print("  jax offset mode=cpu_origin (JAX patch with CPU origin)")
                if args.offset_mode == "cpu_hires":
                    print("  jax offset mode=cpu_hires (CPU hires -> JAX downsample+FFT)")
                if args.offset_mode == "jax_hires":
                    print(
                        "  jax offset mode=jax_hires (JAX hires -> CPU downsample path) "
                        f"shift=({args.jax_hires_shift[0]:+.3f},{args.jax_hires_shift[1]:+.3f}) "
                        f"sign={args.jax_hires_sign}"
                    )

            if cpu_img is None and args.cpu_dir:
                cpu_path = os.path.join(args.cpu_dir, f"cpu_{cutout_index}.npy")
                if os.path.exists(cpu_path):
                    cpu_img = np.load(cpu_path)

            if cpu_img is not None and jax_img is not None:
                diff = jax_img - cpu_img
                diff_l2 = np.sqrt(np.mean(diff**2))
                diff_l1 = np.mean(np.abs(diff))
                print(f"  diff: L1={diff_l1:.6g} L2={diff_l2:.6g}")

            if args.save_dir:
                if cpu_img is not None:
                    np.save(os.path.join(args.save_dir, f"cpu_{cutout_index}.npy"), cpu_img)
                if jax_img is not None:
                    np.save(os.path.join(args.save_dir, f"jax_{cutout_index}.npy"), jax_img)
                if cpu_img is not None and jax_img is not None:
                    np.save(os.path.join(args.save_dir, f"diff_{cutout_index}.npy"), diff)


if __name__ == "__main__":
    main()
