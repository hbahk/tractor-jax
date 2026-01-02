from __future__ import print_function
import numpy as np

from tractor.utils import BaseParams, ParamList, MultiParams, ArithmeticParams
from tractor import ducks


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
        from astrometry.util.starutil_numpy import degrees_between
        return degrees_between(self.ra, self.dec, pos.ra, pos.dec)
