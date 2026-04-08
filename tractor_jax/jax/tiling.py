import numpy as np
import math
from tractor_jax.image import Image
from tractor_jax.sky import ConstantSky

def tile_image(image, tile_size, halo):
    """
    Splits an image into tiles with halo padding.

    Args:
        image: tractor.Image object.
        tile_size: int, size of the core tile (e.g. 256).
        halo: int, padding to add on each side.

    Returns:
        List of (tile_image, metadata).
        metadata is a dict containing:
            'x0', 'y0': Origin of the tile core in original image.
            'core_w', 'core_h': Size of the tile core.
            'pad_left', 'pad_right', 'pad_top', 'pad_bottom': Padding applied.
            'x_start', 'y_start', 'x_end', 'y_end': Extent in original image.
    """
    H, W = image.shape
    tiles = []

    # Grid of tiles
    nx = int(math.ceil(W / tile_size))
    ny = int(math.ceil(H / tile_size))

    # Original Data
    data = image.getImage()
    invvar = image.getInvError()**2

    # WCS
    wcs = image.getWcs()

    # PSF
    psf = image.getPsf()

    # Sky
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

            # Slicing with padding
            # Calculate overlap with image
            im_x0 = max(0, x_start)
            im_y0 = max(0, y_start)
            im_x1 = min(W, x_end)
            im_y1 = min(H, y_end)

            # Slice size
            tile_h = y_end - y_start
            tile_w = x_end - x_start

            tile_data = np.zeros((tile_h, tile_w), dtype=data.dtype)
            tile_invvar = np.zeros((tile_h, tile_w), dtype=invvar.dtype)

            # Offsets in tile
            t_x0 = im_x0 - x_start
            t_y0 = im_y0 - y_start
            t_x1 = t_x0 + (im_x1 - im_x0)
            t_y1 = t_y0 + (im_y1 - im_y0)

            if im_x1 > im_x0 and im_y1 > im_y0:
                tile_data[t_y0:t_y1, t_x0:t_x1] = data[im_y0:im_y1, im_x0:im_x1]
                tile_invvar[t_y0:t_y1, t_x0:t_x1] = invvar[im_y0:im_y1, im_x0:im_x1]

            # Construct new WCS
            # Shifted subtracts offsets from CRPIX, effectively moving the origin.
            # New pixel (0,0) corresponds to Old pixel (x_start, y_start).
            tile_wcs = wcs.shifted(x_start, y_start)

            # Construct Tile Image
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
    """
    Projects all sources in the catalog to pixel coordinates using the given WCS.

    Returns:
        numpy array of shape (N, 2) containing (x, y) pixel coordinates.
        Rows corresponding to sources that failed projection will have NaN.
    """
    positions = []
    # Try to verify if we can vectorize
    # But catalog is list of objects.

    for src in catalog:
        try:
            x, y = wcs.positionToPixel(src.getPosition(), src)
            positions.append([x, y])
        except:
            positions.append([np.nan, np.nan])

    return np.array(positions)

def filter_sources_by_box(positions, x_min, x_max, y_min, y_max, margin=0):
    """
    Returns indices of positions that fall within the box [x_min, x_max) x [y_min, y_max)
    padded by margin.
    """
    x = positions[:, 0]
    y = positions[:, 1]

    # Handle NaNs (sources that failed projection) -> False
    # Use range with margin

    mask = (x >= x_min - margin) & (x < x_max + margin) & \
           (y >= y_min - margin) & (y < y_max + margin)

    indices = np.where(mask)[0]
    return indices
