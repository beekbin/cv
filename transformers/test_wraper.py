from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import time

import transform as tr
import chain
import utils

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def getChain3():
    worker = chain.Chain("chain3")

    resizer = chain.ResizeWraper(w=480, h=480, chance=1.0)
    worker.addOperator(resizer)

    #fmodel = "./model/haarcascade_frontalface_default.xml"
    #facer = chain.FaceWraper(fmodel, chance=1.0)
    #worker.addOperator(facer)

    eraser = chain.EraseWraper(chance=1.0)
    worker.addOperator(eraser)

    noiser = chain.NoiseWraper(chance=1.0, maxSigma=2)
    worker.addOperator(noiser)

    aspect = chain.AspectWraper(chance=1.0)
    worker.addOperator(aspect)

    shrink = chain.ShrinkWraper(chance=1.0, fac_low=0.8)
    worker.addOperator(shrink)

    x = (-20, 20)
    y = (-20, 20)
    z = (-20, 20)
    rotate3d = chain.Rotate3DWraper(chance=1.0,x=x, y=y, z=z)
    worker.addOperator(rotate3d)

    shadow = chain.ShadowWraper(chance=1.0)
    worker.addOperator(shadow)
    return worker


def getChain2():
    worker = chain.Chain("chain2")

    resizer = chain.ResizeWraper(w=480, h=480, chance=1.0)
    worker.addOperator(resizer)

    #fmodel = "./model/haarcascade_frontalface_default.xml"
    #facer = chain.FaceWraper(fmodel, chance=1.0)
    #worker.addOperator(facer)

    shadow = chain.ShadowWraper(chance=0.5)
    worker.addOperator(shadow)

    noiser = chain.NoiseWraper(chance=0.5, maxSigma=6)
    worker.addOperator(noiser) 

    eraser = chain.EraseWraper(chance=1.0)
    worker.addOperator(eraser)

    rotate3d = chain.Rotate3DWraper(chance=1.0)
    worker.addOperator(rotate3d)
    return worker


def getChain1():
    worker = chain.Chain("chain1")

    fmodel = "./model/haarcascade_frontalface_default.xml"
    facer = chain.FaceWraper(fmodel, chance=1.0)
    worker.addOperator(facer)

    resizer = chain.ResizeWraper(w=480, h=480, chance=1.0)
    worker.addOperator(resizer)

    noiser = chain.NoiseWraper(chance=1.0, maxSigma=6)
    worker.addOperator(noiser)

    color = chain.ColorWraper(chance=1.0)
    worker.addOperator(color)

    eraser = chain.EraseWraper(chance=1.0)
    worker.addOperator(eraser)

    rotate3d = chain.Rotate3DWraper(chance=1.0)
    worker.addOperator(rotate3d)
    return worker


def testPerformance(imgs):
    worker = getChain1()
    log.info("%s" % (worker))

    start = time.time()
    for _ in range(32):
        for img in imgs:
            img2 = worker.run(img)

        #img = tr.resize(img, (480, 480))
        #tr.showImgs([img, img2])
    delta = time.time() - start
    log.debug("time used: %s" % (delta))
    return img2


def testEffect(imgs):
    worker = getChain1()
    log.info("%s" % (worker))
    for i in range(len(imgs)):
        img = imgs[i]
        img2 = worker.run(img)
        img = tr.resize(img, (480, 480))
        tr.showImgs([img, img2])
        #tr.saveImgs([img, img2], "./result/%d.jpg"%(i))
    return


def tranImgs(imgs):
    noise = chain.NoiseWraper(0.5, maxSigma=5)
    fmodel = "./model/haarcascade_frontalface_default.xml"
    face = chain.FaceWraper(fmodel, 1.0)
    color = chain.ColorWraper(0.5)
    aspect = chain.AspectWraper(1.0)
    shadow = chain.ShadowWraper(1.0)

    shrink = chain.ShrinkWraper(chance=1.0)
    rotate2D = chain.Rotate2DWraper(chance=1.0, angle=(-30, 30))
    rotate2DX = chain.Rotate2DXWraper(chance=1.0, angle=(-0.5, 0.5))

    rotate3D = chain.Rotate3DWraper(chance=1.0)
    rotate3DX = chain.Rotate3DXWraper(chance=1.0)

    croper = chain.EraseWraper(chance=1.0)

    print(noise)
    print(color)
    for i in range(len(imgs)):
        img = imgs[i]

        img2 = img.copy()
        img = tr.resize(img, (480, 360))

        img2 = face.run(img2)
        img2 = tr.resize(img2, (480, 360))
        #img2 = eraser.run(img2)
        img2 = croper.run(img2)
        #img2 = rotate2DX.run(img2)
        img2 = rotate3D.run(img2)
        #img2 = noise.run(img2)

        #img2 = color.run(img2)
        #img2  shadow.run(img2)
        #img2 = aspect.run(img2)
        #img2 = shrink.run(img2)
        tr.showImgs([img, img2])
        #tr.saveImgs([img, img2], "./result/%d.jpg"%(i))
    return


def testImgs():
    fnames = utils.getFileNames("../data/driver/")
    imgs = []
    for fname in fnames:
        img = tr.loadImage(fname)
        imgs.append(img)

    log.debug("%d imgs." % (len(imgs)))
    testEffect(imgs)
    #testPerformance(imgs)
    return


def main():
    #test()
    testImgs()
    return


if __name__ == "__main__":
    utils.setupLog()
    main()
