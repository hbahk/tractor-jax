from __future__ import print_function
import numpy as np

from tractor.utils import BaseParams, ParamList, MultiParams, ArithmeticParams
from tractor import ducks

try:
    from astropy.wcs import WCS as AstropyWCSObject
    from astropy import wcs as astropy_wcs
    _have_astropy = True
except ImportError:
    _have_astropy = False

class NullWCS(BaseParams, ducks.WCS):
    '''
    The "identity" WCS -- useful when you are using raw pixel
    positions rather than RA,Decs.
    '''

    def __init__(self, pixscale=1., dx=0., dy=0.):
        '''
        pixscale: [arcsec/pix]
        '''
        self.dx = dx
        self.dy = dy
        self.setPixscale(pixscale)

    def hashkey(self):
        return ('NullWCS', self.dx, self.dy)

    def setPixscale(self, pixscale):
        self.pixscale = pixscale
        self.cd = np.array([[1., 0.], [0., 1.]]) *  self.pixscale / 3600.
        self.cd_inverse = np.array([[1., 0.], [0., 1.]]) / (self.pixscale / 3600.)

    def positionToPixel(self, pos, src=None):
        return pos.x + self.dx, pos.y + self.dy

    def pixelToPosition(self, x, y, src=None):
        return x - self.dx, y - self.dy

    def cdAtPixel(self, x, y):
        return self.cd

    def cdInverseAtPixel(self, x, y):
        return self.cd_inverse

    def cdInverseAtPosition(self, pos, src=None):
        return self.cd_inverse

    def pixscale_at(self, x, y):
        return self.pixscale

    def shifted(self, x, y):
        return self.copy()


class AstropyWCS(BaseParams, ducks.WCS):
    '''
    A Tractor WCS implementation that wraps an astropy.wcs.WCS object.
    '''
    def __init__(self, wcs, origin=0):
        '''
        wcs: an astropy.wcs.WCS object.
        origin: 0 or 1.
          Tractor uses 0-based pixel coordinates (center of first pixel is 0,0).
          FITS uses 1-based (center of first pixel is 1,1).
          If `origin` is 0, input (x,y) to positionToPixel/pixelToPosition
          are treated as 0-based (numpy-like) coordinates.
        '''
        if not _have_astropy:
            raise ImportError("Astropy is required to use AstropyWCS")
        self.wcs = wcs
        self.origin = origin

    def hashkey(self):
        # WCS objects are mutable and don't hash nicely; use the header string representation.
        return ('AstropyWCS', self.wcs.to_header_string(), self.origin)

    def positionToPixel(self, pos, src=None):
        if hasattr(pos, 'ra') and hasattr(pos, 'dec'):
            ra, dec = pos.ra, pos.dec
        elif hasattr(pos, 'x') and hasattr(pos, 'y'):
             # Already pixel? This shouldn't happen for a celestial WCS usually.
             return pos.x, pos.y
        else:
             raise ValueError("Unknown position type: %s" % type(pos))

        # astropy.wcs.all_world2pix handles scalar inputs
        x, y = self.wcs.all_world2pix(ra, dec, self.origin)
        return float(x), float(y)

    def pixelToPosition(self, x, y, src=None):
        ra, dec = self.wcs.all_pix2world(x, y, self.origin)
        return RaDecPos(ra, dec)

    def cdAtPixel(self, x, y):
        '''
        Returns the CD matrix at pixel x,y:
        [ [ dRA/dx * cos(Dec), dRA/dy * cos(Dec) ],
          [ dDec/dx          , dDec/dy           ] ]
        '''
        # We compute derivatives numerically to handle distortions correctly.
        delta = 1e-4
        r0, d0 = self.wcs.all_pix2world(x, y, self.origin)
        r1, d1 = self.wcs.all_pix2world(x + delta, y, self.origin)
        r2, d2 = self.wcs.all_pix2world(x, y + delta, self.origin)

        # Derivatives of RA, Dec with respect to x, y
        dRa_dx = (r1 - r0) / delta
        dDec_dx = (d1 - d0) / delta
        dRa_dy = (r2 - r0) / delta
        dDec_dy = (d2 - d0) / delta

        cosdec = np.cos(np.deg2rad(d0))

        cd = np.array([
            [dRa_dx * cosdec, dRa_dy * cosdec],
            [dDec_dx        , dDec_dy        ]
        ])
        return cd

    def pixscale_at(self, x, y):
        # Return sqrt(det(CD matrix)) * 3600
        cd = self.cdAtPixel(x, y)
        det = np.abs(np.linalg.det(cd))
        return 3600. * np.sqrt(det)

    def shifted(self, dx, dy):
        # Create a copy and shift the CRPIX.
        # CRPIX is the pixel coordinate of the reference point.
        # If we shift the image window by (dx, dy),
        # a pixel (x,y) in the new image corresponds to (x+dx, y+dy) in the original.
        # The WCS should map (0,0) in new image to same sky as (dx, dy) in old image.
        # Old WCS: P_old -> Sky.  P_old = P_new + shift.
        # New WCS should map P_new -> Sky = WCS(P_new + shift).
        # Typically this is done by modifying CRPIX.
        # CRPIX_new = CRPIX_old - shift.

        wcs_copy = self.wcs.deepcopy()
        # wcs.wcs.crpix is usually 1-based (FITS convention) but stored as array.
        # Subtracting shifts works regardless of origin if we are just shifting the grid.
        wcs_copy.wcs.crpix[0] -= dx
        wcs_copy.wcs.crpix[1] -= dy

        # Also need to handle SIP distortions if present?
        # If SIP is present, crpix is used in the linear transformation before SIP.
        # However, SIP coefficients depend on pixel coordinates relative to CRPIX.
        # If we change CRPIX, we are changing where the origin of SIP polynomial is?
        # Wait. SIP is usually defined relative to CRPIX.
        # If we change CRPIX, we effectively move the reference point on the image.
        # This is correct for a crop/shift. The "physical" reference point stays at the same sky location,
        # but its pixel coordinate in the new image is different.

        return AstropyWCS(wcs_copy, origin=self.origin)


class PixPos(ParamList):
    '''
    A Position implementation using pixel positions.
    '''
    @staticmethod
    def getNamedParams():
        return dict(x=0, y=1)

    def __init__(self, *args):
        super(PixPos, self).__init__(*args)
        self.stepsizes = [0.1, 0.1]
        self.maxstep = [1., 1.]

    def __str__(self):
        return 'pixel (%.2f, %.2f)' % (self.x, self.y)

    def getDimension(self):
        return 2

class RaDecPos(ArithmeticParams, ParamList):
    '''
    A Position implementation using RA,Dec positions, in degrees.

    Attributes:
      * ``.ra``
      * ``.dec``
    '''
    @staticmethod
    def getName():
        return "RaDecPos"

    @staticmethod
    def getNamedParams():
        return dict(ra=0, dec=1)

    def __str__(self):
        return '%s: RA, Dec = (%.5f, %.5f)' % (self.getName(), self.ra, self.dec)

    def __init__(self, *args, **kwargs):
        super(RaDecPos, self).__init__(*args, **kwargs)
        # self.setStepSizes(1e-4)
        delta = 1e-4
        self.setStepSizes([delta / np.cos(np.deg2rad(self.dec)), delta])

    def getDimension(self):
        return 2
    # def setStepSizes(self, delta):
    #    self.stepsizes = [delta / np.cos(np.deg2rad(self.dec)),delta]

    def distanceFrom(self, pos):
        # from astrometry.util.starutil_numpy import degrees_between
        # Using simple approximation or implementing it here if astrometry is missing
        # But tractor.wcs previously imported it.
        # Let's implement a simple great circle distance
        ra1, dec1 = np.deg2rad(self.ra), np.deg2rad(self.dec)
        ra2, dec2 = np.deg2rad(pos.ra), np.deg2rad(pos.dec)
        d = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
        # clip to avoid domain errors
        d = np.clip(d, -1.0, 1.0)
        return np.rad2deg(np.arccos(d))

class AffineWCS(BaseParams, ducks.WCS):
    '''
    A WCS implementation using a simple affine transformation (CD matrix).
    Supports JAX-traceable operations if attributes are arrays.
    '''
    def __init__(self, crpix, crval, cd):
        self.crpix = crpix
        self.crval = crval
        self.cd = cd
        try:
            self.cd_inv = np.linalg.inv(self.cd)
        except Exception:
            # Handle JAX Tracers
            import jax.numpy as jnp
            self.cd_inv = jnp.linalg.inv(self.cd)

    def hashkey(self):
        return ('AffineWCS', tuple(self.crpix), tuple(self.crval), tuple(self.cd.ravel()))

    def positionToPixel(self, pos, src=None):
        if hasattr(pos, 'x') and hasattr(pos, 'y'):
             return pos.x, pos.y

        # Tangent plane projection for RaDecPos
        ra, dec = pos.ra, pos.dec
        ra0, dec0 = self.crval

        d2r = np.pi / 180.0
        ra_r, dec_r = ra * d2r, dec * d2r
        ra0_r, dec0_r = ra0 * d2r, dec0 * d2r

        xi = np.cos(dec_r) * np.sin(ra_r - ra0_r)
        eta = np.sin(dec_r) * np.cos(dec0_r) - np.cos(dec_r) * np.sin(dec0_r) * np.cos(ra_r - ra0_r)

        r2d = 180.0 / np.pi
        xi_deg = xi * r2d
        eta_deg = eta * r2d

        # (pix - crpix) = CD_inv * (xi, eta)
        # Using einsum or dot. If mapped, need jnp.
        # But this class is used in JAX context if registered.
        # If running in JAX, self.cd_inv is a Tracer.
        # np.dot handles tracers usually if using jax.numpy?
        # But this file imports numpy as np.
        # If inputs are tracers, dispatch happens.

        uv = np.array([xi_deg, eta_deg])

        # Handle batching?
        # If pos is scalar (one source), uv is (2,). cd_inv is (2,2).
        # np.dot(cd_inv, uv) -> (2,)

        # Note: If running inside vmap, np.* functions might be JAX primitives if jax.numpy is imported as np,
        # but here it is standard numpy.
        # However, JAX tracers passed to numpy functions often trigger JAX dispatch or error.
        # Safe way is to use operators.

        # xy = cd_inv @ uv + crpix
        # But standard numpy doesn't support @ for Tracers seamlessly if linalg involved?
        # Actually JAX tracers implement __array_ufunc__ / __matmul__.

        xy = self.cd_inv @ uv + self.crpix

        return xy[0], xy[1]

    def pixelToPosition(self, x, y, src=None):
        xy = np.array([x, y]) - self.crpix
        uv = self.cd @ xy
        xi_deg, eta_deg = uv[0], uv[1]

        # Simple approx inverse
        ra0, dec0 = self.crval
        return RaDecPos(ra0 + xi_deg / np.cos(np.deg2rad(dec0)), dec0 + eta_deg)

    def cdAtPixel(self, x, y):
        return self.cd

    def cdInverseAtPixel(self, x, y):
        return self.cd_inv

    def pixscale_at(self, x, y):
        det = np.abs(np.linalg.det(self.cd))
        return 3600. * np.sqrt(det)

    def shifted(self, dx, dy):
        return AffineWCS(self.crpix - np.array([dx, dy]), self.crval, self.cd)
