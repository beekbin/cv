from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import transform as tr
import chain

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def setupLog():
    """
    Reset logging: to make the logging.basicConfig() work again.

    Step 1: remove all the root.handlers
       logging.basicConfig() does nothing if the root logger already has handlers
            configured for it.
    Step 2: setup logging format
        logging.basicConfig() will create and add a default handler to the root logger.
    """
    if logging.root:
        logging.root.handlers = []
    fmt = "%(levelname).3s[%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    dtfmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=dtfmt)
    return


def tranImgs(imgs):
    noise = chain.NoiseWraper(0.5, maxSigma=5)
    color = chain.ColorWraper(0.5)
    aspect = chain.AspectWraper(0.5)
    shadow = chain.ShadowWraper(1.0)
    print(noise)
    print(color)
    for img in imgs:
        img = tr.resize(img, (480, 360))
        img2 = noise.run(img)
        #img2 = color.run(img2)
        img2 = shadow.run(img2)
        img2 = aspect.run(img2)
        tr.showImgs([img, img2])

    return


def testImgs():
    fnames = [
         "../data/driver/songbin.png",
         "../data/driver/songbin2.png",
         "../data/driver/motorist.jpg",
         "../data/driver/minnesota.png",
         "../data/driver/california.png",
         "../data/driver/ny2.png",
         "../data/driver/missi.png",
         "../data/driver/vermont.png",
         "../data/driver/florida.jpg",
         "../data/driver/ma.jpg",
    ]
    imgs = []
    for fname in fnames:
        img = tr.loadImage(fname)
        imgs.append(img)

    tranImgs(imgs)
    return


def main():
    #test()
    testImgs()
    return


if __name__ == "__main__":
    setupLog()
    main()
