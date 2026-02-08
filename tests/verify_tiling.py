import numpy as np
import jax
from tractor.jax.optimizer import optimize_fluxes
from tractor.engine import Tractor
from tractor.image import Image
from tractor.pointsource import PointSource
from tractor.wcs import NullWCS, PixPos
from tractor.brightness import Flux
from tractor.psf import PixelizedPSF

def test_tiling_optimization():
    print("Testing tiling optimization...")

    # Create a 512x512 image
    H, W = 512, 512
    data = np.zeros((H, W), dtype=np.float32)
    inverr = np.ones((H, W), dtype=np.float32)
    wcs = NullWCS()

    # Create PSF
    psf_img = np.zeros((11, 11), dtype=np.float32)
    psf_img[5, 5] = 1.0
    psf = PixelizedPSF(psf_img)

    img = Image(data=data, inverr=inverr, wcs=wcs, psf=psf)

    # Sources
    # 1. At (100, 100) (Tile 0,0)
    # 2. At (400, 400) (Tile 1,1 if tile_size=256)
    src1 = PointSource(PixPos(100, 100), Flux(100.))
    src2 = PointSource(PixPos(400, 400), Flux(200.))

    catalog = [src1, src2]
    tractor = Tractor([img], catalog)

    # Run optimization with tiling
    print("Running with tiling...")
    # tile_size=256 -> 4 tiles (2x2)
    # src1 should be in tile 0. src2 in tile 3.
    # Other tiles empty.

    results = optimize_fluxes(
        tractor,
        use_tiling=True,
        tile_size=256,
        return_variances=False
    )

    print("Optimization results (list of fluxes per tile):")
    print(results)

    # results is list of length N_tiles = 4.
    assert len(results) == 4

    # Check results
    res0 = results[0] # Tile 0
    print("Tile 0 fluxes:", res0)
    # Src1 (idx 0) should change. Src2 (idx 1) should stay 100/200.

    assert abs(res0[0]) < 1.0 # Optimized towards 0
    assert abs(res0[1] - 200.0) < 1e-5 # Unchanged

    res3 = results[3] # Tile 3 (400, 400)
    print("Tile 3 fluxes:", res3)
    # Src1 (idx 0) unchanged. Src2 (idx 1) optimized.
    assert abs(res3[0] - 100.0) < 1e-5
    assert abs(res3[1]) < 1.0

    print("Test passed.")

if __name__ == "__main__":
    test_tiling_optimization()
