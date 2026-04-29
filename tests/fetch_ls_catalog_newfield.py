"""Fetch the Legacy Survey DR10 catalog for a SPHEREx target field.

Mirrors the testphot.ipynb workflow (cells 13-15) but parameterised so it can be
pointed at a new field. Defaults are set to the new SPHEREx test field at
RA=347.0925, Dec=-2.1921 (matches `testdata/spherex_cutouts/summary.ecsv`).

Authenticates against NOIRLab Data Lab (`dl`) and submits an async query to
ls_dr10.tractor.

Example
-------
    python tests/fetch_ls_catalog_newfield.py \
        --ra 347.0925 --dec -2.1921 --name newfield_a347d2

The query radius defaults to half the diagonal of a 100x100 native-pixel cutout
(6.15"/pix), so every catalog source that could possibly fall inside any cutout
in the field is included.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from getpass import getpass
from pathlib import Path

import numpy as np
from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert

logger = logging.getLogger(__name__)

DEFAULT_COLNAMES = [
    "ls_id", "type", "ra", "dec",
    "flux_g", "flux_ivar_g",
    "flux_r", "flux_ivar_r",
    "flux_i", "flux_ivar_i",
    "flux_z", "flux_ivar_z",
    "flux_w1", "flux_ivar_w1",
    "flux_w2", "flux_ivar_w2",
    "mag_g", "mag_r", "mag_i", "mag_z",
    "sersic", "shape_r", "shape_e1", "shape_e2",
    "dered_flux_g", "dered_flux_r", "dered_flux_i", "dered_flux_z",
    "dered_flux_w1", "dered_flux_w2",
    "g_r", "r_z", "z_w1",
]

# SPHEREx native pixel scale, arcsec.
SPHEREX_PIXSCALE = 6.15
# Default native pixel size of a cutout (matches the new spherex-retrieval format).
DEFAULT_CUTOUT_PIX = 100


def submit_query(name: str, ra_deg: float, dec_deg: float, radius_deg: float,
                 colnames=DEFAULT_COLNAMES) -> str:
    cols = "*" if colnames in ("*", None) else ", ".join(colnames)
    sql = (
        f"SELECT {cols} FROM ls_dr10.tractor "
        f"WHERE q3c_radial_query(ra, dec, {ra_deg}, {dec_deg}, {radius_deg})"
    )
    out = f"vos://tmp/ls_{name}.csv"
    logger.info("Submitting query for %s (radius=%.4f deg)", name, radius_deg)
    jobid = qc.query(sql=sql, out=out, async_=True)
    logger.info("Job ID: %s", jobid)
    return jobid


def wait_for_results(jobid: str, poll_seconds: float = 5.0,
                     timeout_seconds: float = 1800.0):
    t0 = time.time()
    while True:
        st = qc.status(jobid)
        if st == "COMPLETED":
            res = qc.results(jobid)
            tab = convert(res, "table")
            for col in tab.colnames:
                if tab[col].dtype == "float64":
                    tab[col] = tab[col].astype("float32")
            return tab
        if st == "ERROR":
            try:
                err = qc.error(jobid)
            except Exception:
                err = "<no error message>"
            raise RuntimeError(f"Datalab query failed (job={jobid}): {err}")
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"Datalab query {jobid} did not finish in "
                               f"{timeout_seconds:.0f} s (last status={st})")
        time.sleep(poll_seconds)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, default=347.0925, help="field center RA (deg)")
    p.add_argument("--dec", type=float, default=-2.1921, help="field center Dec (deg)")
    p.add_argument("--cutout-pixels", type=int, default=DEFAULT_CUTOUT_PIX,
                   help="native pixels along one side of the cutout")
    p.add_argument("--radius-arcsec", type=float, default=None,
                   help="override query radius in arcsec (default: cutout half-diagonal)")
    p.add_argument("--name", default="newfield",
                   help="short name used in the temporary VOSpace path and output filename")
    p.add_argument("--out", default=None,
                   help="output parquet path (default: tests/ls_<name>.parquet)")
    p.add_argument("--user", default=os.environ.get("DATALAB_USER"),
                   help="Datalab username (or set DATALAB_USER)")
    p.add_argument("--password-env", default="DATALAB_PASSWORD",
                   help="env var name to read the password from (skips getpass prompt)")
    p.add_argument("--columns", default=None,
                   help="comma-separated subset of column names; default uses the script's set")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.radius_arcsec is None:
        # Half-diagonal of the cutout, so that any source potentially landing
        # in any cutout in this field is captured.
        radius_arcsec = 0.5 * math.sqrt(2) * args.cutout_pixels * SPHEREX_PIXSCALE
    else:
        radius_arcsec = args.radius_arcsec
    radius_deg = radius_arcsec / 3600.0

    user = args.user or input("Datalab username: ")
    pwd = os.environ.get(args.password_env) or getpass("Datalab password: ")
    ac.login(user, pwd)
    logger.info("Logged into Datalab as %s", ac.whoAmI())

    cols = (args.columns.split(",") if args.columns else DEFAULT_COLNAMES)
    jobid = submit_query(args.name, args.ra, args.dec, radius_deg, colnames=cols)
    tab = wait_for_results(jobid)
    n_before = len(tab)
    tab = tab[tab["type"] != "DUP"]
    logger.info("Retrieved %d rows (%d after dropping DUP)", n_before, len(tab))

    out = Path(args.out) if args.out else Path(__file__).with_name(f"ls_{args.name}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.write(str(out), overwrite=True)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
