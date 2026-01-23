import numpy as np
import re
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import conf
from scipy.interpolate import InterpolatedUnivariateSpline
from tractor import mixture_profiles as mp
from tractor.sersic import SersicMixture, SersicGalaxy

XX, YY = np.meshgrid(np.arange(11), np.arange(11))
ZXGRID = XX.flatten() + 1
ZYGRID = YY.flatten() + 1


def get_psf_zone_table(psf_header):
    """
    Extract the PSF zone table from the PSF extension header.
    """
    xctr = {}
    yctr = {}

    for key, val in psf_header.items():
        # Look for keys like XCTR* or YCTR*
        xm = re.match(r"(XCTR*)", key)
        if xm:
            xplane = int(key.split("_")[1])
            xctr[xplane] = val
        ym = re.match(r"(YCTR*)", key)
        if ym:
            yplane = int(key.split("_")[1])
            yctr[yplane] = val

    assert len(xctr) == len(yctr)

    tab = Table(names=["zone_id", "x", "y"], dtype=[int, float, float])
    for zone_id in xctr.keys():
        tab.add_row([zone_id, xctr[zone_id], yctr[zone_id]])

    return tab


def get_nearest_psf_zone_index(x, y, psf_zone_table):
    dist = np.sqrt((psf_zone_table["x"] - x) ** 2 + (psf_zone_table["y"] - y) ** 2)
    return psf_zone_table["zone_id"][np.argmin(dist)]



def process_cutout(row, ra, dec, cache, timeout=3600):
    """
    Downloads the cutouts given in a row of the table including all SPHEREx images overlapping with a position.

    Parameters:
    ===========

    row : astropy.table row
        Row of a table that will be changed in place by this function. The table
        is created by the SQL TAP query.
    ra,dec : coordinates (astropy units)
        Ra and Dec coordinates (same as used for the TAP query) with attached astropy units
    cache : bool
        If set to `True`, the output of cached and the cutout processing will run faster next time.
        Turn this feature off by setting `cache = False`.
    timeout : float
        Timeout for the remote request in seconds.
    """
    with conf.set_temp('remote_timeout', timeout):
        with fits.open(row["uri"], cache=cache) as hdulist:
            # There are seven HDUs:
            # 0 contains minimal metadata in the header and no data.
            # 1 through 6 are: IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE
            header = hdulist[1].header

            # Compute pixel coordinates corresponding to cutout position.
            spatial_wcs = WCS(header)
            x, y = spatial_wcs.world_to_pixel(
                SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
            )

            # Compute the original pixel positions
            crpix1a = header["CRPIX1A"]
            crpix2a = header["CRPIX2A"]
            x_orig = 1 + (x - crpix1a)
            y_orig = 1 + (y - crpix2a)
            row["X"] = x_orig
            row["Y"] = y_orig

            # save the detector array number
            row["DETECTOR"] = hdulist[1].header["DETECTOR"]

            # Compute wavelength at cutout position.
            spectral_wcs = WCS(header, fobj=hdulist, key="W")
            spectral_wcs.sip = None
            wavelength, bandpass = spectral_wcs.pixel_to_world(x, y)
            row["central_wavelength"] = wavelength.to(u.micrometer).value

            # Collect the HDUs for this cutout and append the row's cutout_index to the EXTNAME.
            hdus = []
            for hdu in hdulist[
                1:5
            ]:  # skip the primary header and append until the ZODI extension
                hdu.header["EXTNAME"] = f"{hdu.header['EXTNAME']}{row['cutout_index']}"
                hdus.append(
                    hdu.copy()
                )  # Copy so the data is available after the file is closed

            # get the PSF cube only for the PSF zones that overlap with the cutout
            psf_zone_table = get_psf_zone_table(hdulist[5].header)
            xlo, xhi = 0, hdulist[1].header["NAXIS1"]
            ylo, yhi = 0, hdulist[1].header["NAXIS2"]
            xlo_orig, xhi_orig = 1 + (xlo - crpix1a), 1 + (xhi - crpix1a)
            ylo_orig, yhi_orig = 1 + (ylo - crpix2a), 1 + (yhi - crpix2a)

            # get the PSF zone coords for the lower left and upper right corners of the cutout
            zid_ll = get_nearest_psf_zone_index(xlo_orig, ylo_orig, psf_zone_table)
            zid_ur = get_nearest_psf_zone_index(xhi_orig, yhi_orig, psf_zone_table)
            zx_ll, zy_ll = ZXGRID[zid_ll - 1], ZYGRID[zid_ll - 1]
            zx_ur, zy_ur = ZXGRID[zid_ur - 1], ZYGRID[zid_ur - 1]

            # get the zone indices for the overlapping zones
            sel = (
                (ZXGRID >= zx_ll)
                & (ZXGRID <= zx_ur)
                & (ZYGRID >= zy_ll)
                & (ZYGRID <= zy_ur)
            )

            # get the PSF cube for the overlapping zones
            psf_cube = hdulist[5].data[sel, :, :]

            # replace the PSF extension with the new PSF cube and append the new PSF extension
            hdulist[5].data = psf_cube
            hdulist[5].header["NAXIS3"] = psf_cube.shape[0]

            hdulist[5].header["EXTNAME"] = f"PSF{row['cutout_index']}"
            hdus.append(hdulist[5].copy())

            # save lookup table for the PSF zone indices
            cols = fits.ColDefs(
                [
                    fits.Column(
                        name="zone_id",
                        format="J",
                        array=psf_zone_table["zone_id"][sel],
                        unit="",
                    ),
                    fits.Column(
                        name="x", format="D", array=psf_zone_table["x"][sel], unit=""
                    ),
                    fits.Column(
                        name="y", format="D", array=psf_zone_table["y"][sel], unit=""
                    ),
                ]
            )
            psf_zone_table_hdu = fits.BinTableHDU.from_columns(cols)
            psf_zone_table_hdu.header["EXTNAME"] = f"PSF_ZONE_LOOKUP{row['cutout_index']}"
            hdus.append(psf_zone_table_hdu)

            row["hdus"] = hdus


def sky_pa_to_pixel_pa(wcs, ra_deg, dec_deg, pa_sky_deg, d_arcsec=1.0, y_down=False):
    """
    Convert a position angle measured on the sky (East of North basis: +x=East, +y=North)
    to a position angle in the image pixel frame (+x=right, +y=up by default).

    Parameters
    ----------
    wcs : astropy.wcs.WCS (celestial)
    ra_deg, dec_deg : float
        Target sky position in degrees (ICRS).
    pa_sky_deg : float
        Position angle on the sky in degrees, measured from +RA(East) toward +Dec(North).
        (e.g., phi = 0.5*atan2(e2,e1) in degrees)
    d_arcsec : float
        Small step used to probe local Jacobian (default 1").
    y_down : bool
        If your image/model uses +y downward (e.g., some rendering or libraries),
        set True to flip the pixel y-axis.

    Returns
    -------
    pa_pix_deg : float
        Position angle in pixel frame, measured from +x (right) toward +y (up unless y_down=True).
    """
    sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")

    # Base pixel
    x0, y0 = wcs.world_to_pixel(sc)

    # Unit steps on sky: +East (RA increases), +North (Dec increases)
    d = (d_arcsec * u.arcsec).to(u.deg).value
    sc_E = SkyCoord(ra=(ra_deg + d/np.cos(np.deg2rad(dec_deg)))*u.deg, dec=dec_deg*u.deg, frame="icrs")
    sc_N = SkyCoord(ra=ra_deg*u.deg, dec=(dec_deg + d)*u.deg, frame="icrs")

    xE, yE = wcs.world_to_pixel(sc_E)
    xN, yN = wcs.world_to_pixel(sc_N)

    # Pixel vectors for +East and +North at (ra,dec)
    vE = np.array([xE - x0, yE - y0])
    vN = np.array([xN - x0, yN - y0])

    # Optional: flip pixel y if your consumer uses y-down convention
    if y_down:
        vE[1] *= -1.0
        vN[1] *= -1.0

    # Sky direction unit vector in (East, North)
    th = np.deg2rad(pa_sky_deg)
    d_sky = np.array([np.cos(th), np.sin(th)])

    # Map to pixel plane via local 2x2 transform [vE vN]
    M = np.column_stack([vE, vN])  # columns are East and North in pixel basis
    v_pix = M @ d_sky

    pa_pix = np.rad2deg(np.arctan2(v_pix[1], v_pix[0]))
    return pa_pix

            
class SPHERExSersicMixture(SersicMixture):
    singleton = None

    @staticmethod
    def getProfile(sindex):
        if SPHERExSersicMixture.singleton is None:
            SPHERExSersicMixture.singleton = SPHERExSersicMixture()
        return SPHERExSersicMixture.singleton._getProfile(sindex)

    def __init__(self):
        super().__init__()

        self.beyonds = [
            (0.29, -0.00844581249119647),
            (0.3, -0.007543589678601026),
            (0.31, -0.006720049694663777),
            (0.32, -0.0059662304247363185),
            (0.33, -0.005277313728958344),
            (0.34, -0.0046521468417414225),
            (0.35, -0.004095346709762526),
            (0.36, -0.0035988688279965375),
            (0.37, -0.0031498118023494115),
            (0.38, -0.0027583846582809324),
            (0.39, -0.002435351519089801),
            (0.4, -0.002157335977535313),
            (0.41, -0.0019076077741795316),
            (0.42, -0.0016991406639236262),
            (0.43, -0.001425029716449977),
            (0.44, -0.0012650990437518272),
            (0.45, -0.0011347785247703968),
            (0.46, -0.0009879528471214982),
            (0.47, -0.0008100151316562387),
            (0.48, -0.0005870125324175524),
            (0.49, -0.0003190764835062643),
            (0.5, 5.425903805811316e-06),
            (0.51, 0.00034705152982811294),
            (0.515, 0.0002996544861531003),
            (0.52, 0.00022447215877752225),
            (0.53, 0.00010594007072395328),
            (0.54, 0.0002694537629742144),
            (0.55, 0.0004300827684946551),
            (0.56, 0.0005944090931850887),
            (0.57, 0.0015975318101883462),
            (0.575, 0.0007583228570509637),
            (0.58, 0.0006995684031628757),
            (0.6, 0.0007162051886123733),
            (0.62, 0.0010120879777582026),
            (0.63, 0.0011267421968184088),
            (0.64, 0.0012801213226446007),
            (0.65, 0.001451646001103979),
            (0.7, 0.002508762419644761),
            (0.71, 0.002766796388286308),
            (0.72, 0.003023229882584244),
            (0.73, 0.0032915776909705485),
            (0.74, 0.003561967457950399),
            (0.75, 0.0038673532336104266),
            (0.8, 0.005528522125680335),
            (0.85, 0.007468553253350441),
            (0.9, 0.00965566942039231),
            (0.95, 0.01205320678893651),
            (1.0, 0.014659549792626902),
            (1.1, 0.017005210259315284),
            (1.2, 0.01927687860355387),
            (1.3, 0.021484300887706198),
            (1.4, 0.02363493914344006),
            (1.5, 0.025774453598841895),
            (1.55, 0.02653204147973942),
            (1.6, 0.027731530277451344),
            (1.7, 0.02997295724425736),
            (1.8, 0.031100832053232386),
            (1.9, 0.03353732320270597),
            (2.0, 0.03415553715599429),
            (2.1, 0.035866666758461396),
            (2.3, 0.03925398489658294),
            (2.5, 0.04261982372191825),
            (2.7, 0.045970050960330966),
            (3.0, 0.05097239750541088),
            (3.1, 0.05244179457571285),
            (3.2, 0.05267681959158704),
            (3.3, 0.051008278225892156),
            (3.4, 0.051581162287590465),
            (3.5, 0.052756205813684454),
            (4.0, 0.05858779172595929),
            (4.5, 0.07168299370139825),
            (5.0, 0.0841544214197475),
            (5.5, 0.09593308483544921),
            (6.0, 0.10697422593718764),
            (6.1, 0.10906657469929443),
            (6.2, 0.11117858609339415),
            (6.3, 0.1132645336487178),
        ]

        self.cores = [
            (0.29, -0.0006145669842789747),
            (0.3, -0.00047558548715020965),
            (0.31, -0.000363217224507717),
            (0.32, -0.0002860576454500885),
            (0.33, -0.0002047657848697204),
            (0.34, -0.00014498433184317872),
            (0.35, -0.00010651925123739137),
            (0.36, -8.396446125114032e-05),
            (0.37, -4.507138441234293e-05),
            (0.38, -3.380973759070649e-05),
            (0.39, -2.060885174259841e-05),
            (0.4, -1.7333510825889853e-05),
            (0.41, -1.3655362836484386e-05),
            (0.42, -2.2053831621571263e-05),
            (0.43, 3.519140248531283e-05),
            (0.44, 5.8527269772845614e-05),
            (0.45, 6.389492410741049e-05),
            (0.46, 5.951319146363376e-05),
            (0.47, 4.41916821859456e-05),
            (0.48, 2.961055933942136e-05),
            (0.49, 1.2627912384211015e-05),
            (0.5, 1.212783262150019e-07),
            (0.51, -1.9430637892225988e-05),
            (0.515, -4.212502048273059e-05),
            (0.52, -3.228734261651045e-05),
            (0.53, 7.81361399720959e-06),
            (0.54, -4.824476926845733e-07),
            (0.55, -7.352795657555866e-06),
            (0.56, -1.4297605179125483e-05),
            (0.57, -0.0008618049018462859),
            (0.575, -3.926782323415701e-06),
            (0.58, 1.862366342647581e-05),
            (0.6, 2.3501590432239983e-05),
            (0.62, 3.091940991745146e-05),
            (0.63, 1.3872692724903324e-05),
            (0.64, 8.008823279725963e-06),
            (0.65, 5.0818015268627725e-06),
            (0.7, 1.0753942212227141e-05),
            (0.71, 1.3588086665461407e-05),
            (0.72, 1.233305641651361e-05),
            (0.73, 1.1090788702317056e-05),
            (0.74, 8.221140030351126e-06),
            (0.75, 8.823047828843134e-06),
            (0.8, 1.2454565203434687e-05),
            (0.85, 1.5852421938133965e-05),
            (0.9, 2.027803646037496e-05),
            (0.95, 2.3860516997376013e-05),
            (1.0, 3.39957344719366e-05),
            (1.1, 5.54545242191784e-05),
            (1.2, 8.690513039832926e-05),
            (1.3, 0.0001277037233881062),
            (1.4, 0.0001860448049312291),
            (1.5, 0.0002658230042170695),
            (1.55, 0.00021350097227901266),
            (1.6, 0.00031839967024277493),
            (1.7, 0.0004857433537968636),
            (1.8, 0.00037686920762880494),
            (1.9, 0.0006261207533730384),
            (2.0, 0.0005666093530467542),
            (2.1, 0.0007145367012874604),
            (2.3, 0.0010893594944085816),
            (2.5, 0.0015779745432438763),
            (2.7, 0.00218967829239175),
            (3.0, 0.0033474709444939466),
            (3.1, 0.0037814734649312953),
            (3.2, 0.004189334066699302),
            (3.3, 0.0048892121748450035),
            (3.4, 0.00551991679573588),
            (3.5, 0.0060998023127743495),
            (4.0, 0.009448723687390248),
            (4.5, 0.012080413858493733),
            (5.0, 0.01499344940562819),
            (5.5, 0.018145661429931625),
            (6.0, 0.021495074522919932),
            (6.1, 0.022184349060213604),
            (6.2, 0.02288044574498377),
            (6.3, 0.02358058762144588),
        ]

        self.core_func = InterpolatedUnivariateSpline(
            [s for s, c in self.cores], [c for s, c in self.cores], k=3
        )
        self.beyond_func = InterpolatedUnivariateSpline(
            [s for s, b in self.beyonds], [b for s, b in self.beyonds], k=3
        )

    def _getProfile(self, sindex):
        matches = []
        # clamp
        if sindex <= self.lowest:
            matches.append(self.fits[0])
            sindex = self.lowest
        elif sindex >= self.highest:
            matches.append(self.fits[-1])
            sindex = self.highest
        else:
            for f in self.fits:
                lo, hi, a, v = f
                if sindex >= lo and sindex < hi:
                    matches.append(f)

        if len(matches) == 2:
            # Two ranges overlap.  Ramp between them.
            # Assume self.fits is ordered in increasing Sersic index
            m0, m1 = matches
            lo0, hi0, a0, v0 = m0
            lo1, hi1, a1, v1 = m1
            assert lo0 < lo1
            assert lo1 < hi0  # overlap is in here
            ramp_lo = lo1
            ramp_hi = hi0
            assert ramp_lo < ramp_hi
            assert ramp_lo <= sindex
            assert sindex < ramp_hi
            ramp_frac = (sindex - ramp_lo) / (ramp_hi - ramp_lo)
            amps0 = np.array([f(sindex) for f in a0])
            amps0 /= amps0.sum()
            amps1 = np.array([f(sindex) for f in a1])
            amps1 /= amps1.sum()
            amps = np.append((1.0 - ramp_frac) * amps0, ramp_frac * amps1)
            varr = np.exp(np.array([f(sindex) for f in v0 + v1]))
        else:
            assert len(matches) == 1
            lo, hi, amp_funcs, logvar_funcs = matches[0]
            amps = np.array([f(sindex) for f in amp_funcs])
            amps /= amps.sum()
            varr = np.exp(np.array([f(sindex) for f in logvar_funcs]))

        # Core
        core = self.core_func(sindex)
        beyond = self.beyond_func(sindex)
        amps *= (1.0 - core - beyond) / amps.sum()

        amps = np.append(amps, core)
        varr = np.append(varr, 0.0)

        return mp.MixtureOfGaussians(amps, np.zeros((len(amps), 2)), varr)


class SPHERExSersicGalaxy(SersicGalaxy):
    def getProfile(self):
        return SPHERExSersicMixture.getProfile(self.sersicindex.val)