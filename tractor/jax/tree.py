import jax
from tractor.engine import Tractor, Images, Catalog
from tractor.image import Image
from tractor.pointsource import PointSource
from tractor.sky import ConstantSky, NullSky
from tractor.wcs import NullWCS, PixPos, RaDecPos
from tractor.psf import PixelizedPSF, GaussianMixturePSF
from tractor.utils import MogParams, ScalarParam, ParamList
from tractor.brightness import Flux, LinearPhotoCal, NullPhotoCal

def register_pytree_nodes():
    # Tractor
    jax.tree_util.register_pytree_node(
        Tractor,
        lambda t: ((t.images, t.catalog), (t.optimizer, t.model_kwargs)),
        lambda aux, children: Tractor(images=children[0], catalog=children[1],
                                      optimizer=aux[0], model_kwargs=aux[1])
    )

    # Images (MultiParams)
    jax.tree_util.register_pytree_node(
        Images,
        lambda i: ((i.subs,), (i.liquid,)),
        lambda aux, children: _restore_multiparams(Images, children[0], aux[0])
    )

    # Catalog (MultiParams)
    jax.tree_util.register_pytree_node(
        Catalog,
        lambda c: ((c.subs,), (c.liquid,)),
        lambda aux, children: _restore_multiparams(Catalog, children[0], aux[0])
    )

    # Image
    jax.tree_util.register_pytree_node(
        Image,
        lambda im: ((im.data, im.inverr, im.psf, im.wcs, im.sky, im.photocal),
                    (im.name, im.time, im.zr, im.liquid)),
        lambda aux, children: _restore_image(children, aux)
    )

    # PointSource
    jax.tree_util.register_pytree_node(
        PointSource,
        lambda ps: ((ps.pos, ps.brightness), (ps.fixedRadius, ps.minRadius, ps.liquid)),
        lambda aux, children: _restore_pointsource(children, aux)
    )

    # ConstantSky (ScalarParam)
    jax.tree_util.register_pytree_node(
        ConstantSky,
        lambda s: ((s.val,), None),
        lambda aux, children: ConstantSky(children[0])
    )

    # NullSky
    jax.tree_util.register_pytree_node(
        NullSky,
        lambda s: ((), None),
        lambda aux, children: NullSky()
    )

    # NullWCS
    jax.tree_util.register_pytree_node(
        NullWCS,
        lambda w: ((w.pixscale, w.dx, w.dy), None),
        lambda aux, children: NullWCS(children[0], dx=children[1], dy=children[2])
    )

    # PixelizedPSF
    jax.tree_util.register_pytree_node(
        PixelizedPSF,
        lambda p: ((p.img,), (p.sampling, p.Lorder)),
        lambda aux, children: PixelizedPSF(children[0], sampling=aux[0], Lorder=aux[1])
    )

    # GaussianMixturePSF (MogParams)
    jax.tree_util.register_pytree_node(
        GaussianMixturePSF,
        lambda p: ((p.mog.amp, p.mog.mean, p.mog.var), (getattr(p, 'radius', None),)),
        lambda aux, children: _restore_gmpsf(children, aux)
    )

    # MogParams
    jax.tree_util.register_pytree_node(
        MogParams,
        lambda p: ((p.mog.amp, p.mog.mean, p.mog.var), None),
        lambda aux, children: MogParams(children[0], children[1], children[2])
    )

    # ParamList and subclasses
    # We must register subclasses explicitly to preserve type
    for cls in [ParamList, PixPos, RaDecPos]:
        jax.tree_util.register_pytree_node(
            cls,
            lambda p: ((p.vals,), (p.liquid,)),
            lambda aux, children, cls=cls: _restore_paramlist(cls, children[0], aux[0])
        )

    # ScalarParam and subclasses
    for cls in [ScalarParam, Flux, LinearPhotoCal]:
        jax.tree_util.register_pytree_node(
            cls,
            lambda p: ((p.val,), None),
            lambda aux, children, cls=cls: cls(children[0])
        )

    # NullPhotoCal
    jax.tree_util.register_pytree_node(
        NullPhotoCal,
        lambda p: ((), None),
        lambda aux, children: NullPhotoCal()
    )


def _restore_multiparams(cls, subs, liquid):
    obj = cls(*subs)
    obj.liquid = liquid
    return obj

def _restore_image(children, aux):
    data, inverr, psf, wcs, sky, photocal = children
    name, time, zr, liquid = aux
    img = Image(data=data, inverr=inverr, psf=psf, wcs=wcs, sky=sky, photocal=photocal,
                name=name, time=time, zr=zr)
    img.liquid = liquid
    return img

def _restore_pointsource(children, aux):
    pos, brightness = children
    fixedRadius, minRadius, liquid = aux
    ps = PointSource(pos, brightness)
    ps.fixedRadius = fixedRadius
    ps.minRadius = minRadius
    ps.liquid = liquid
    return ps

def _restore_gmpsf(children, aux):
    amp, mean, var = children
    radius, = aux
    psf = GaussianMixturePSF(amp, mean, var)
    if radius is not None:
        psf.radius = radius
    return psf

def _restore_paramlist(cls, vals, liquid):
    pl = cls(*vals)
    pl.liquid = liquid
    return pl
