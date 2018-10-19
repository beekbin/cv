from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import transform as tr

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


def tranImgs(imgs):
    for img in imgs:
        img = tr.resize(img, (480, 360))
        #img2 = tr.addNoise(img, sigma=5.0)
        #img2 = tr.adjustBrightness(img, 0.8)
        #img2 = tr.adjustContrast(img, 2.0)
        #img2 = tr.adjustSaturation(img, 4.0)
        #img2 = tr.adjustHue(img, 0.2)
        #img2 = tr.adjustGamma(img, 1.5)
        
        #img2 = tr.gradualShade(img, 0.9, 1, 0)
        #img2 = tr.gradualShadeV(img, 0.9, 1)

        img2 = tr.simpleRotate(img, 30)
        #img2 = tr.rotateX(img, 0.98)

        #img2 = tr.crop(img, (1.0, 0.3), point=(0.0, 0.0))
        #img2 = tr.adjustAspectRatio(img, 2.0)
        #img2 = tr.adjustPerspectiveX(img, anglex=30, angley=-45, anglez=-30, shear=8)
        #img2 = tr.adjustPerspective(ig)
        tr.showImgs([img, img2])
    return

def test():
    #fname = "../data/driver/songbin2.png"
    fname = "../data/driver/california.png"
    img = tr.loadImage(fname)
    #tr.showImg(img, "Songbin")

    #img2 = tr.addNoise(img)
    #img2 = tr.adjustBrightness(img, 0.8)
    #img2 = tr.adjustContrast(img, 2.0)
    #img2 = tr.adjustSaturation(img, 4.0)
    #img2 = tr.adjustHue(img, 0.2)
    #img2 = tr.adjustGamma(img, 1.5)
    #img2 = tr.simpleRotate(img, 8)
    img2 = tr.gradualShade(img, 1.1)
    tr.showImgs([img, img2])
    return


def main():
    #test()
    testImgs()
    return


if __name__ == "__main__":
    setupLog()
    main()

