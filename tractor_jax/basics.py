"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`basics.py`
===========

Generally useful generic implementations of things the Tractor needs:
Magnitudes, (RA,Dec) positions, FITS WCS, and so on.

"""

import numpy as np

from tractor_jax.image import Image
from tractor_jax.patch import Patch
from tractor_jax.utils import MultiParams
from tractor_jax import mixture_profiles as mp

from tractor_jax.tractortime import TAITime
from tractor_jax.psf import (PixelizedPSF, GaussianMixturePSF,
                         GaussianMixtureEllipsePSF, NCircularGaussianPSF)
from tractor_jax.wcs import (NullWCS, PixPos, RaDecPos)
from tractor_jax.sky import NullSky, ConstantSky
from tractor_jax.brightness import (Mag, Flux, Mags, Fluxes, NanoMaggies,
                                FluxesPhotoCal, MagsPhotoCal, NullPhotoCal,
                                LinearPhotoCal)
from tractor_jax.pointsource import BasicSource, SingleProfileSource, PointSource
from tractor_jax.shifted import (ParamsWrapper, ShiftedPsf, ScaledPhotoCal,
                             ScaledWcs, ShiftedWcs)


class TractorWCSWrapper(object):
    '''
    Wraps a Tractor WCS object to look like an
    astrometry.util.util.Tan/Sip object.
    '''

    def __init__(self, wcs, w, h, x0=0, y0=0):
        self.wcs = wcs
        self.imagew = w
        self.imageh = h
        self.x0 = x0
        self.y0 = y0

    def pixelxy2radec(self, x, y):
        # FITS pixels x,y
        rd = self.wcs.pixelToPosition(x + self.x0 - 1, y + self.y0 - 1)
        return rd.ra, rd.dec

    def radec2pixelxy(self, ra, dec):
        # Vectorized?
        if hasattr(ra, '__len__') or hasattr(dec, '__len__'):
            try:
                b = np.broadcast(ra, dec)
                ok = np.ones(b.shape, bool)
                x = np.zeros(b.shape)
                y = np.zeros(b.shape)
                rd = RaDecPos(0., 0.)
                for i, (r, d) in enumerate(b):
                    rd.ra = r
                    rd.dec = d
                    xi, yi = self.wcs.positionToPixel(rd)
                    x.flat[i] = xi
                    y.flat[i] = yi
                x += 1 - self.x0
                y += 1 - self.y0
                return ok, x, y
            except:
                pass

        x, y = self.wcs.positionToPixel(RaDecPos(ra, dec))
        return True, x - self.x0 + 1, y - self.y0 + 1


def getParamTypeTree(param):
    mytype = type(param)
    if isinstance(param, MultiParams):
        return [mytype] + [getParamTypeTree(s) for s in param._getActiveSubs()]
    return [mytype]
