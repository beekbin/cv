from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import utils
import transform as tr

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def testImgs():
    #fnames = utils.getFileNames("../data/driver/")
    fnames = utils.getFileNames("../data/ca/")
    imgs = []
    for fname in fnames:
        img = tr.loadImage(fname)
        imgs.append(img)

    tranImgs(imgs)   
    return


def tranImgs(imgs):
    for i in range(len(imgs)):
        img = imgs[i]
        img = tr.resize(img, (600, 400))
        #img2 = tr.addNoise(img, sigma=5.0)
        img2 = img.copy()
        fac = 2.5
        log.info("fac = %.3f" % (fac))
        #img2 = tr.adjustBrightness(img2, fac)
        #img2 = tr.adjustContrast(img2, fac)
        img2 = tr.adjustSaturation(img, fac)
        #img2 = tr.adjustHue(img, 0.2)
        #img2 = tr.adjustGamma(img2, 1.4)
    
        #img2 = tr.gradualShade(img, 0.9, 1, 0)
        #img2 = tr.gradualShadeV(img, 0.9, 1)

        #img2 = tr.rotate2D(img2, 30)
        #img2 = tr.rotate2DX(img, 0.98)

        #img2 = tr.crop(img2, (1.0, 0.3), point=(0.0, 0.0))
        #img2 = tr.adjustAspectRatio(img, 2.0)
        #scale=(1.0, 1.0)
        #translate = (5, 7)
        #img2 = tr.adjustPerspective(img2, anglex=0, angley=0, anglez=15, shear=0, scale=scale)
        #img3 = tr.adjustPerspective(img, anglex=30, angley=0, anglez=45, shear=0)
        #img2 = tr.adjustPerspectiveX(img2)
        tr.showImgs([img, img2])
        #tr.saveImgs([img, img2], "./result/%d.jpg" % (i))
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
    #img2 = tr.rotate2D(img, 8)
    img2 = tr.gradualShade(img, 1.1)
    tr.showImgs([img, img2])
    return


def main():
    #test()
    testImgs()
    return


if __name__ == "__main__":
    utils.setupLog()
    main()

