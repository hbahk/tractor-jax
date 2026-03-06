import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table
from utils import get_nearest_psf_zone_index


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


def compute_psf_stats(psf):
    h, w = psf.shape
    cy, cx = h // 2, w // 2
    max_pos = np.unravel_index(np.argmax(psf), psf.shape)
    yy, xx = np.indices(psf.shape)
    total = psf.sum()
    if total == 0:
        centroid = (np.nan, np.nan)
    else:
        centroid = (np.sum(yy * psf) / total, np.sum(xx * psf) / total)
    return {
        "shape": (h, w),
        "center_pix": (cy, cx),
        "max_pos": max_pos,
        "centroid": centroid,
        "max_offset": (max_pos[0] - cy, max_pos[1] - cx),
        "centroid_offset": (centroid[0] - cy, centroid[1] - cx),
        "sum": total,
    }


def summarize(label, stats):
    h, w = stats["shape"]
    cy, cx = stats["center_pix"]
    my, mx = stats["max_pos"]
    cty, ctx = stats["centroid"]
    dy_m, dx_m = stats["max_offset"]
    dy_c, dx_c = stats["centroid_offset"]
    return (
        f"{label} shape={h}x{w} center=({cy},{cx}) "
        f"max=({my},{mx}) dmax=({dy_m:+.3f},{dx_m:+.3f}) "
        f"centroid=({cty:.3f},{ctx:.3f}) dcentroid=({dy_c:+.3f},{dx_c:+.3f}) "
        f"sum={stats['sum']:.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits",
        default="tests/testphot.fits",
        help="Path to testphot.fits",
    )
    parser.add_argument(
        "--cutouts",
        type=int,
        nargs="+",
        default=[127, 174, 402, 2796, 2854, 2924],
        help="Cutout indices (1-based, matching IMAGE{cutout_index})",
    )
    parser.add_argument(
        "--show-downsampled",
        action="store_true",
        help="Also report stats after downsample_psf_oversample2",
    )
    args = parser.parse_args()

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

            # cutout_info rows are 0-based; cutout_index is 1-based
            row = cutout_info[cutout_index - 1]
            gx, gy = row["x"], row["y"]
            zoneid = get_nearest_psf_zone_index(gx, gy, psf_lookup)
            zidx = np.where(psf_lookup["zone_id"] == zoneid)[0][0]
            psf = psf_cube[zidx]

            stats = compute_psf_stats(psf)
            print(f"\n[{cutout_index}] zone_id={zoneid} zidx={zidx}")
            print("  " + summarize("raw", stats))

            if args.show_downsampled:
                psf_ds = downsample_psf_oversample2(psf)
                stats_ds = compute_psf_stats(psf_ds)
                print("  " + summarize("down2x", stats_ds))


if __name__ == "__main__":
    main()
