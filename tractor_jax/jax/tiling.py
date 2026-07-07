import numpy as np
import math
from tractor_jax.image import Image
from tractor_jax.sky import ConstantSky

def tile_image(image, tile_size, halo):
    """Split an image into tiles with halo padding.

    Covers the image with a grid of core tiles of size ``tile_size``,
    each extended by ``halo`` pixels on every side. Regions of the halo
    that fall outside the original image are zero-padded (data and
    inverse variance). Each tile carries a shifted WCS whose pixel
    ``(0, 0)`` corresponds to pixel ``(x_start, y_start)`` of the
    original image; the PSF and sky models are shared with the parent
    image.

    Parameters
    ----------
    image : tractor_jax.image.Image
        Input image object.
    tile_size : int
        Size of the core tile in pixels (e.g. 256).
    halo : int
        Padding to add on each side, in pixels.

    Returns
    -------
    list of tuple
        List of ``(tile_image, metadata)`` pairs. ``tile_image`` is a
        ``tractor_jax.image.Image``; ``metadata`` is a dict containing:

        - ``'x0'``, ``'y0'`` : origin of the tile core in the original
          image.
        - ``'core_w'``, ``'core_h'`` : size of the tile core.
        - ``'halo'`` : the halo padding applied on each side.
        - ``'x_start'``, ``'y_start'``, ``'x_end'``, ``'y_end'`` :
          extent of the padded tile in original-image coordinates.
    """
    H, W = image.shape
    tiles = []

    nx = int(math.ceil(W / tile_size))
    ny = int(math.ceil(H / tile_size))

    data = image.getImage()
    invvar = image.getInvError()**2

    wcs = image.getWcs()
    psf = image.getPsf()
    sky = image.getSky()

    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_size
            y0 = iy * tile_size

            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)

            core_w = x1 - x0
            core_h = y1 - y0

            x_start = x0 - halo
            y_start = y0 - halo
            x_end = x1 + halo
            y_end = y1 + halo

            # Overlap of the padded tile with the image; the rest stays zero
            im_x0 = max(0, x_start)
            im_y0 = max(0, y_start)
            im_x1 = min(W, x_end)
            im_y1 = min(H, y_end)

            tile_h = y_end - y_start
            tile_w = x_end - x_start

            tile_data = np.zeros((tile_h, tile_w), dtype=data.dtype)
            tile_invvar = np.zeros((tile_h, tile_w), dtype=invvar.dtype)

            # Offsets of the overlap region within the tile
            t_x0 = im_x0 - x_start
            t_y0 = im_y0 - y_start
            t_x1 = t_x0 + (im_x1 - im_x0)
            t_y1 = t_y0 + (im_y1 - im_y0)

            if im_x1 > im_x0 and im_y1 > im_y0:
                tile_data[t_y0:t_y1, t_x0:t_x1] = data[im_y0:im_y1, im_x0:im_x1]
                tile_invvar[t_y0:t_y1, t_x0:t_x1] = invvar[im_y0:im_y1, im_x0:im_x1]

            # shifted() subtracts offsets from CRPIX, moving the origin:
            # new pixel (0,0) corresponds to old pixel (x_start, y_start).
            tile_wcs = wcs.shifted(x_start, y_start)

            tile_inverr = np.sqrt(tile_invvar)

            tile_img = Image(data=tile_data, inverr=tile_inverr, wcs=tile_wcs, psf=psf, sky=sky)
            tile_img.name = f"{getattr(image, 'name', 'img')}_tile_{ix}_{iy}"

            meta = {
                'x0': x0, 'y0': y0,
                'core_w': core_w, 'core_h': core_h,
                'halo': halo,
                'x_start': x_start, 'y_start': y_start,
                'x_end': x_end, 'y_end': y_end
            }
            tiles.append((tile_img, meta))

    return tiles

def project_catalog(catalog, wcs):
    """Project all catalog sources to pixel coordinates with a given WCS.

    Parameters
    ----------
    catalog : iterable
        Iterable of source objects providing ``getPosition()``.
    wcs
        WCS object providing ``positionToPixel(position, source)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 2)`` containing ``(x, y)`` pixel
        coordinates. Rows corresponding to sources that failed
        projection are set to NaN.
    """
    positions = []
    # Catalog is a list of heterogeneous objects; loop rather than vectorize.
    for src in catalog:
        try:
            x, y = wcs.positionToPixel(src.getPosition(), src)
            positions.append([x, y])
        except:
            positions.append([np.nan, np.nan])

    return np.array(positions)

def filter_sources_by_box(positions, x_min, x_max, y_min, y_max, margin=0):
    """Select positions falling within a (margin-padded) bounding box.

    Returns the indices of positions inside the half-open box
    ``[x_min - margin, x_max + margin) x [y_min - margin, y_max + margin)``.
    Positions containing NaN (sources that failed projection) are
    excluded.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape ``(N, 2)`` of ``(x, y)`` pixel coordinates.
    x_min : float
        Lower x bound of the box.
    x_max : float
        Upper x bound of the box.
    y_min : float
        Lower y bound of the box.
    y_max : float
        Upper y bound of the box.
    margin : float, optional
        Padding applied to each side of the box (default 0).

    Returns
    -------
    numpy.ndarray
        Integer indices of the positions inside the padded box.
    """
    x = positions[:, 0]
    y = positions[:, 1]

    # NaN positions (failed projections) compare False and are excluded
    mask = (x >= x_min - margin) & (x < x_max + margin) & \
           (y >= y_min - margin) & (y < y_max + margin)

    indices = np.where(mask)[0]
    return indices
