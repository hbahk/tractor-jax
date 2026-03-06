import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table


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


def compute_metrics(img, flg, var, bkg):
    invvar = np.zeros_like(var, dtype=np.float64)
    good_var = (var > 0) & np.isfinite(var)
    invvar[good_var] = 1.0 / var[good_var]

    mask = (flg & MASKBITS) != 0
    valid = (~mask) & np.isfinite(img) & np.isfinite(bkg) & np.isfinite(invvar) & (invvar > 0)

    if np.any(valid):
        data = img - bkg
        snr = np.abs(data) * np.sqrt(invvar)
        snr_valid = snr[valid]
        data_valid = data[valid]
    else:
        snr_valid = np.array([])
        data_valid = np.array([])

    metrics = {
        "mask_fraction": float(mask.mean()),
        "valid_fraction": float(valid.mean()),
        "bkg_median": float(np.nanmedian(bkg)),
        "bkg_mean": float(np.nanmean(bkg)),
        "bkg_std": float(np.nanstd(bkg)),
        "data_median": float(np.nanmedian(data_valid)) if data_valid.size else np.nan,
        "data_std": float(np.nanstd(data_valid)) if data_valid.size else np.nan,
        "snr_median": float(np.nanmedian(snr_valid)) if snr_valid.size else np.nan,
        "snr_mean": float(np.nanmean(snr_valid)) if snr_valid.size else np.nan,
    }
    return metrics


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
    args = parser.parse_args()

    with fits.open(args.fits, memmap=True) as hdul:
        _ = Table(hdul[1].data)
        for cutout_index in args.cutouts:
            img_hdu = f"IMAGE{cutout_index}"
            flg_hdu = f"FLAGS{cutout_index}"
            var_hdu = f"VARIANCE{cutout_index}"
            bkg_hdu = f"ZODI{cutout_index}"

            if any(h not in hdul for h in [img_hdu, flg_hdu, var_hdu, bkg_hdu]):
                print(f"[{cutout_index}] missing HDU(s), skipping")
                continue

            img = hdul[img_hdu].data
            flg = hdul[flg_hdu].data
            var = hdul[var_hdu].data
            bkg = hdul[bkg_hdu].data

            m = compute_metrics(img, flg, var, bkg)
            print(
                f"[{cutout_index}] mask_frac={m['mask_fraction']:.3f} "
                f"valid_frac={m['valid_fraction']:.3f} "
                f"bkg_med={m['bkg_median']:.4g} bkg_std={m['bkg_std']:.4g} "
                f"data_med={m['data_median']:.4g} data_std={m['data_std']:.4g} "
                f"snr_med={m['snr_median']:.4g} snr_mean={m['snr_mean']:.4g}"
            )


if __name__ == "__main__":
    main()
