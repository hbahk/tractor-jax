"""Duck-type definitions of types used by the Tractor.

Most of this code is not actually used at all.  It's here for
documentation purposes.

Notes
-----
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.
"""


class Params(object):
    '''A set of parameters that can be optimized by the Tractor.

    This is a duck-type definition.
    '''

    def copy(self):
        return None

    def hashkey(self):
        '''Return a tuple containing the state of this `Params` object
        for use as a cache key.

        Returns
        -------
        tuple
            The state of this object.  All elements must be hashable:
            see http://docs.python.org/glossary.html#term-hashable
        '''
        return ()

    # def __hash__(self):
    #    ''' Params must be hashable. '''
    #    return None
    # def __eq__(self, other):

    def getParamNames(self):
        '''Return the names of the parameters.

        Returns
        -------
        list of str
            The names of the parameters.
        '''
        return []

    def numberOfParams(self):
        '''Return the number of parameters (ie, number of scalar values).

        Returns
        -------
        int
            The number of scalar parameter values.
        '''
        return len(self.getParams())

    def getParams(self):
        '''Return a *copy* of the current parameter values.

        Returns
        -------
        iterable
            A copy of the current parameter values (eg, a list).
        '''
        return []

    def getAllParams(self):
        return self.getParams()

    def getAllStepSizes(self, *args, **kwargs):
        return self.getStepSizes(*args, **kwargs)

    def getStepSizes(self, *args, **kwargs):
        '''Return "reasonable" step sizes for the parameters.

        Returns
        -------
        list of float
            One step size per parameter.
        '''
        return []

    def setAllStepSizes(self, ss):
        self.setStepSizes(ss)

    def setStepSizes(self, ss):
        assert(len(ss) == self.numberOfParams())
        pass

    def setParams(self, p):
        '''Set the parameter values to the values in the given iterable.

        Parameters
        ----------
        p : iterable of float
            The new parameter values.  The length of `p` will be equal
            to ``numberOfParams()``.
        '''
        assert(len(p) == self.numberOfParams())

    def setAllParams(self, p):
        return self.setParams(p)

    def setParam(self, i, p):
        '''Set parameter index `i` to new value `p`.

        Parameters
        ----------
        i : int
            Parameter index, in the range ``[0, numberOfParams())``.
        p : float
            New parameter value.

        Returns
        -------
        float
            The old value of the parameter.
        '''
        return None

    def getLowerBounds(self):
        return []

    def getUpperBounds(self):
        return []

    def getMaxStep(self):
        '''Return the largest step we should take in this parameter.

        Use for nonlinear params where making a large change will take
        us outside the linear optimization regime.

        Returns
        -------
        float or None
            The largest allowed step.
        '''
        return None

    def getGaussianPriors(self):
        '''Return the Gaussian priors on this set of parameters.

        Returns
        -------
        list of tuple
            A list of ``(index, mu, sigma)`` tuples of Gaussian priors
            on this set of parameters.
        '''
        return []

    def getLogPrior(self):
        '''Return the prior, evaluated at the current values of the parameters.

        Returns
        -------
        float
            The log-prior at the current parameter values.
        '''
        return 0.

    def getLogPriorDerivatives(self):
        '''Return a "chi-like" approximation to the log-prior at the
        current parameter values.

        This will go into the least-squares fitting (each term in the
        prior acts like an extra "pixel" in the fit).

        Returns
        -------
        rowA : list of iterables of int
            Row indices describing a sparse matrix ``pA`` of shape
            ``N x numberOfParams``.
        colA : list of int
            Column indices of the sparse matrix ``pA``.
        valA : list of iterables of float
            Values of the sparse matrix ``pA``.
        pb : list of iterables of float
            Right-hand-side terms, of shape ``N``.
        mub : list of iterables of float
            Like `pb` but not shifted relative to the current parameter
            values; ie, it's the mean of the Gaussian.  Shape ``N``.

        Notes
        -----
        ``N`` is the number of "pseudo-pixels" or Gaussian terms.
        ``pA`` will be appended to the least-squares ``A`` matrix, and
        ``pb`` will be appended to the least-squares ``b`` vector, and
        the least-squares problem is minimizing

        .. math:: || A \\cdot (\\mathrm{delta\\text{-}params}) - b ||^2

        This function must take frozen-ness of parameters into account
        (this is implied by the ``numberOfParams`` shape requirement).
        '''
        return None


class ImageCalibration(object):
    def toFitsHeader(self, hdr, prefix=''):
        params = self.getAllParams()
        names = self.getParamNames()
        for i,(name,val) in enumerate(zip(names, params)):
            k = prefix + 'P%i' % i
            hdr.add_record(dict(name=k, value=val, comment=name))

    def toStandardFitsHeader(self, hdr):
        pass

    @classmethod
    def fromFitsHeader(clazz, hdr, prefix=''):
        args = []
        for i in range(100):
            k = prefix + 'A%i' % i
            if not k in hdr:
                break
            args.append(hdr.get(k))
        obj = clazz(*args)
        params = []
        for i in range(100):
            k = prefix + 'P%i' % i
            if not k in hdr:
                break
            params.append(hdr.get(k))
        obj.setAllParams(params)
        return obj


class Sky(ImageCalibration, Params):
    '''Duck-type definition for a sky model.'''

    def getParamDerivatives(self, tractor, img, srcs):
        '''Return the derivatives of this sky model in the given image.

        Parameters
        ----------
        tractor : Tractor
            The Tractor object.
        img : Image
            The image in which to evaluate the derivatives.
        srcs : list of Source
            The sources in the image.

        Returns
        -------
        list of Patch
            ``[ Patch, Patch, ... ]``, of length ``numberOfParams()``,
            containing the derivatives in the given `Image` for each
            parameter.
        '''
        return []

    def addTo(self, mod, scale=1.):
        '''Add the sky to the input synthetic image.

        Parameters
        ----------
        mod : numpy.ndarray
            The 2-D synthetic (model) image to which the sky is added.
        scale : float, optional
            Factor by which to scale the sky before adding.
        '''
        pass

    def getConstant(self):
        '''Return an unspecified constant value, eg the mean, median, etc.

        Returns
        -------
        float
            A constant value characterizing this sky model.
        '''
        return 0.

    def subtract(self, con):
        '''Subtract a constant value from this sky model.

        Parameters
        ----------
        con : float
            The constant value to subtract.
        '''
        raise RuntimeError('Unimplemented: Sky.subtract()')

    def shift(self, x0, y0):
        '''Shift this sky model so that it applies to the subimage
        starting at ``x0, y0``.

        Parameters
        ----------
        x0 : int
            X pixel coordinate of the subimage origin.
        y0 : int
            Y pixel coordinate of the subimage origin.
        '''
        pass

    def shifted(self, x0, y0):
        s = self.copy()
        s.shift(x0, y0)
        return s


class Source(Params):
    '''Duck-type definition of a Source (star, galaxy, etc) that the
    Tractor uses.
    '''

    def getModelPatch(self, img, minsb=0., modelMask=None, **kwargs):
        '''Return a Patch containing a rendering of this Source into
        the given image.

        This will probably use the calibration information of the
        `Image`: the WCS, PSF, and photometric calibration.

        Parameters
        ----------
        img : Image
            The image into which to render this source.
        minsb : float, optional
            The allowable approximation error per pixel; we are asking
            the source to render itself out to this surface brightness.
        modelMask : ModelMask, optional
            Describes the rectangular region of interest (image pixels).

        Returns
        -------
        Patch
            A rendering of this Source into the given `Image` object.
        '''
        pass

    def getParamDerivatives(self, img, modelMask=None, **kwargs):
        '''Return the derivatives of this source in the given image.

        Parameters
        ----------
        img : Image
            The image in which to evaluate the derivatives.
        modelMask : ModelMask, optional
            Describes the rectangular region of interest (image pixels).

        Returns
        -------
        list of Patch
            ``[ Patch, Patch, ... ]``, of length ``numberOfParams()``,
            containing the derivatives in the given `Image` for each
            parameter.
        '''
        return []

    def getBrightnesses(self):
        return []

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None,
                                **kwargs):
        '''Return unit-flux model patches, one per brightness.

        Like ``getModelPatch()``, but ignore the brightness of the
        object and just return a patch whose sum is unity.

        Parameters
        ----------
        img : Image
            The image into which to render this source.
        minval : float, optional
            Like ``minsb``, gives the allowable per-pixel value at
            which the profile can be truncated.  The patch may
            therefore not sum to 1 exactly.
        modelMask : ModelMask, optional
            Describes the rectangular region of interest (image pixels).

        Returns
        -------
        list of Patch
            A list the same length as ``getBrightnesses()``, each
            containing a Patch whose sum is ~ unity.
        '''
        pass


class Brightness(Params):
    '''Duck-type definition of the brightness of an astronomical source.

    Only used as an input to `PhotoCal`.  `Source` objects have
    `Brightness` objects; `PhotoCal` objects convert these into counts
    in a specific `Image`.
    '''
    pass


class PhotoCal(ImageCalibration, Params):
    '''Duck-type definition of photometric calibration.

    A `PhotoCal` belongs to an `Image`; it converts `Brightness`
    values into counts ("data numbers", ADU, etc) in the data space
    (synthetic image) of the `Image`.  It also contains the parameters
    of that conversion so they can be optimized along with everything
    else.

    This relationship need not be linear: the `Brightness` could be an
    astronomical magnitude, for example.  In general, there is a lot
    of freedom in the definition of the `Brightness` object, and
    `PhotoCal` has to be kept consistent with that.
    '''

    def brightnessToCounts(self, brightness):
        '''Convert a brightness into counts.

        Parameters
        ----------
        brightness : Brightness
            A `Brightness` duck to convert.

        Returns
        -------
        float
            The corresponding counts.
        '''
        pass


class Position(Params):
    '''Duck-type definition of the position of an astronomical object.

    Only used as an input to a `WCS` object; `Source` objects have
    `Position` objects, and `WCS` objects convert them into pixel
    coordinates in a specific `Image`.
    '''
    pass


class Time(Params):
    '''Duck-type definition of a time.

    Objects of type `Time` should define arithmetic operators (at least
    ``__sub__``, ``__add__``, ``__isub__``, ``__iadd__``).
    '''
    # def __sub__(self, other):
    #   pass

    def getSunTheta(self):
        '''Return the angle of the Earth's (mean?) anomaly at this time.

        Returns
        -------
        float
            The time of year expressed as an angle in radians.
        '''
        pass

    def toYears(self):
        pass


class WCS(ImageCalibration, Params):
    '''Duck-type definition of World Coordinate System.

    Converts between Position objects and Image pixel coordinates.

    In general, there is a lot of freedom in the definition of the
    `Position` object, and `WCS` has to be kept consistent with that.
    For instance, if the `Position` objects used are image-based x-y
    positions (`PixPos`), then `WCS` has to be null (or close to
    that); `NullWCS`.
    '''

    def positionToPixel(self, pos, src=None):
        '''Convert a position into ``x, y`` pixel coordinates.

        Parameters
        ----------
        pos : Position
            The position to convert.
        src : Source, optional
            The source may be passed in; your `WCS` could have
            color-specific behavior, for example.

        Returns
        -------
        x : float
            X pixel coordinate; ``0, 0`` is the first pixel.
        y : float
            Y pixel coordinate.

        Notes
        -----
        Pixels are funny things.  Our convention is shifted by 1 from
        the FITS convention, so 0,0 is the *center* of the first
        ("zeroth", says Hogg) pixel, if you think of pixels as little
        boxes.  (What is the emoticon for "point and laugh"?)
        '''
        return None

    def cdAtPixel(self, x, y):
        '''Return a local affine relationship between `Position` and
        ``(x, y)`` pixel coordinates.

        This is used, for example, to convert tensor shapes of
        galaxies from `Position` space to image space.

        Parameters
        ----------
        x : float
            X pixel coordinate.
        y : float
            Y pixel coordinate.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2, 2)``, in degrees per pixel.

        Notes
        -----
        In FITS celestial coordinates language, this is the CD matrix
        at pixel x,y::

            [ [ dRA/dx * cos(Dec), dRA/dy * cos(Dec) ],
              [ dDec/dx          , dDec/dy           ] ]

        in FITS these are called::

            [ [ CD11             , CD12              ],
              [ CD21             , CD22              ] ]

        The units of these things are degrees per pixel.
        '''
        return None

    def cdInverseAtPixel(self, x, y):
        import numpy as np
        cd = self.cdAtPixel(x, y)
        cdi = np.linalg.inv(cd)
        return cdi

    def cdInverseAtPosition(self, pos, src=None):
        px, py = self.positionToPixel(pos, src=src)
        return self.cdInverseAtPixel(px, py)

    def pixelDerivsToPositionDerivs(self, pos, src, counts0, patch0, patchdx, patchdy):
        # Convert x,y derivatives to Position derivatives
        cdi = self.cdInverseAtPosition(pos, src=src)
        # Get thawed Position parameter indices
        derivs = []
        for i,pname in pos.getThawedParamIndicesAndNames():
            deriv = (patchdx * cdi[0, i] +
                     patchdy * cdi[1, i]) * counts0
            deriv.setName('d(ptsrc)/d(pos.%s)' % pname)
            derivs.append(deriv)
        return derivs

    def pixscale_at(self, x, y):
        '''Return the local pixel scale at the given pixel coordinates.

        Parameters
        ----------
        x : float
            X pixel coordinate.
        y : float
            Y pixel coordinate.

        Returns
        -------
        float
            The local pixel scale, in *arcseconds* per pixel.
        '''
        import numpy as np
        return 3600. * np.sqrt(np.abs(np.linalg.det(self.cdAtPixel(x, y))))

    def shifted(self, dx, dy):
        '''Return a new WCS object appropriate for a shifted subimage.

        Parameters
        ----------
        dx : float
            X offset of the subimage with respect to the current WCS
            origin.
        dy : float
            Y offset of the subimage.

        Returns
        -------
        WCS
            A new WCS object appropriate for the subimage starting at
            ``(dx, dy)`` with respect to the current WCS origin.
        '''
        return None


class PSF(ImageCalibration, Params):
    '''Duck-type definition of a point-spread function.'''

    def getPointSourcePatch(self, px, py, minval=0., modelMask=None):
        '''Return a rendering of a point source at the given pixel
        coordinates.

        Parameters
        ----------
        px : float
            X pixel coordinate of the point source.
        py : float
            Y pixel coordinate of the point source.
        minval : float, optional
            Says that we are willing to accept an approximation such
            that pixels with counts < `minval` can be omitted.
        modelMask : ModelMask, optional
            Describes the pixels to be evaluated.  If the `modelMask`
            includes a pixel-by-pixel mask, this overrides `minval`.

        Returns
        -------
        Patch
            A rendering of a point source at the given pixel
            coordinates.  The returned `Patch` should have unit
            "counts".
        '''
        pass

    def getRadius(self):
        '''Return the size of the support of this PSF.

        This is required because the Tractor has to decide what size
        to make the ``Patch`` objects.

        Returns
        -------
        float
            The radius of the PSF support, in pixels.
        '''
        return 0

    def getShifted(self, x0, y0):
        '''Return a PSF model for the subimage starting at ``x0, y0``.

        Parameters
        ----------
        x0 : int
            X pixel coordinate of the subimage origin.
        y0 : int
            Y pixel coordinate of the subimage origin.

        Returns
        -------
        PSF
            A PSF model for the subimage.
        '''
        return None

    # Optional: Allows galaxy models to render via analytic convolution:
    # def getMixtureOfGaussians(self, px=None, py=None, **kwargs):
    #     '''
    #     Returns a mixture_profiles.MixtureOfGaussians object approximating this
    #     PSF at the given px,py position.  The mean of the MoG is NOT set to px,py;
    #     it is 0,0.
    #     '''
