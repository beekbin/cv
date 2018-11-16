from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import time

import transform as tr
import wraper
import utils

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

SIZE = (352, 352)

def getChain3():
    worker = wraper.Chain("wraper3")

    resizer = wraper.ResizeWraper(w=SIZE[0], h=SIZE[1], chance=1.0)
    worker.addOperator(resizer)

    #fmodel = "./model/haarcascade_frontalface_default.xml"
    #facer = wraper.FaceWraper(fmodel, chance=1.0)
    #worker.addOperator(facer)

    eraser = wraper.EraseWraper(chance=1.0)
    worker.addOperator(eraser)

    noiser = wraper.NoiseWraper(chance=1.0, maxSigma=2)
    worker.addOperator(noiser)

    aspect = wraper.AspectWraper(chance=1.0)
    worker.addOperator(aspect)

    shrink = wraper.ShrinkWraper(chance=1.0, fac_low=0.8)
    worker.addOperator(shrink)

    x = (-20, 20)
    y = (-20, 20)
    z = (-20, 20)
    rotate3d = wraper.Rotate3DWraper(chance=1.0,x=x, y=y, z=z)
    worker.addOperator(rotate3d)

    shadow = wraper.ShadowWraper(chance=1.0)
    worker.addOperator(shadow)
    return worker


def getChain2():
    worker = wraper.Chain("wraper2")

    resizer = wraper.ResizeWraper(w=SIZE[0], h=SIZE[1], chance=1.0)
    worker.addOperator(resizer)

    #fmodel = "./model/haarcascade_frontalface_default.xml"
    #facer = wraper.FaceWraper(fmodel, chance=1.0)
    #worker.addOperator(facer)

    shadow = wraper.ShadowWraper(chance=0.5)
    worker.addOperator(shadow)

    noiser = wraper.NoiseWraper(chance=0.5, maxSigma=6)
    worker.addOperator(noiser) 

    eraser = wraper.EraseWraper(chance=1.0)
    worker.addOperator(eraser)

    rotate3d = wraper.Rotate3DWraper(chance=1.0)
    worker.addOperator(rotate3d)
    return worker


def getChain1():
    worker = wraper.Chain("wraper1")

    fmodel = "./model/haarcascade_frontalface_default.xml"
    facer = wraper.FaceWraper(fmodel, chance=1.0)
    worker.addOperator(facer)

    eraser = wraper.EraseWraper(chance=1.0)
    worker.addOperator(eraser)

    bg_imgs = "/Users/songbin/dev/data/bg"
    bgwraper = wraper.BackGroundWraper(chance=1.0)
    bgwraper.loadImgs(bg_imgs, (800, 600))
    worker.addOperator(bgwraper)

    #resizer = wraper.ResizeWraper(w=SIZE[0], h=SIZE[1], chance=1.0)
    #worker.addOperator(resizer)

    noiser = wraper.NoiseWraper(chance=1.0, maxSigma=6)
    worker.addOperator(noiser)

    color = wraper.ColorWraper(chance=1.0)
    worker.addOperator(color)


    x = (-20, 20)
    y = (-20, 20)
    z = (-20, 20)
    rotate3d = wraper.Rotate3DWraper(chance=1.0, x=x, y=y, z=z)
    #rotate3d = wraper.Rotate3DWraper(chance=1.0)
    #rotate3d = wraper.Rotate3DXWraper(chance=1.0)
    worker.addOperator(rotate3d)
    return worker


def getChain0():
    worker = wraper.Chain("single worker")

    shrinker = wraper.ShrinkWraper(chance=1.0, fac_low=0.50)
    shrinker.fac_high = 0.52
    worker.addOperator(shrinker)

    resizer = wraper.ResizeWraper(w=SIZE[0], h=SIZE[1], chance=1.0)
    worker.addOperator(resizer)

    return worker


def testPerformance(imgs):
    worker = getChain2()
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
        img = tr.resize(img, SIZE)
        tr.showImgs([img, img2])
        #tr.saveImgs([img, img2], "./result/%d.jpg"%(i))
    return


def tranImgs(imgs):
    noise = wraper.NoiseWraper(0.5, maxSigma=5)
    fmodel = "./model/haarcascade_frontalface_default.xml"
    face = wraper.FaceWraper(fmodel, 1.0)
    color = wraper.ColorWraper(0.5)
    aspect = wraper.AspectWraper(1.0)
    shadow = wraper.ShadowWraper(1.0)

    shrink = wraper.ShrinkWraper(chance=1.0)
    rotate2D = wraper.Rotate2DWraper(chance=1.0, angle=(-30, 30))
    rotate2DX = wraper.Rotate2DXWraper(chance=1.0, angle=(-0.5, 0.5))

    rotate3D = wraper.Rotate3DWraper(chance=1.0)
    rotate3DX = wraper.Rotate3DXWraper(chance=1.0)

    croper = wraper.EraseWraper(chance=1.0)

    print(noise)
    print(color)
    for i in range(len(imgs)):
        img = imgs[i]

        img2 = img.copy()
        img = tr.resize(img, (480, 360))

        img2 = face.run(img2)
        #img2 = tr.resize(img2, (480, 360))
        #img2 = eraser.run(img2)
        #img2 = croper.run(img2)
        #img2 = rotate2DX.run(img2)
        img2 = rotate3DX.run(img2)
        #img2 = noise.run(img2)

        #img2 = color.run(img2)
        #img2  shadow.run(img2)
        #img2 = aspect.run(img2)
        #img2 = shrink.run(img2)
        tr.showImgs([img, img2])
        #tr.saveImgs([img, img2], "./result/%d.jpg"%(i))
    return


def loadImgs(adir):
    """load images from a directory"""
    fnames = utils.getFileNames(adir)
    imgs = []
    for fname in fnames:
        img = tr.loadImage(fname)
        imgs.append(img)

    log.debug("%d imgs." % (len(imgs)))
    return imgs


def testBg():
    #adir = "../data/driver"
    adir = "/Users/songbin/dev/data/docID/small/"
    imgs = loadImgs(adir)
    bg_imgs = "/Users/songbin/dev/data/bg"

    bgwarper = wraper.BackGroundWraper(chance=1.0)
    bgwarper.loadImgs(bg_imgs, (800, 600))

    noiser = wraper.NoiseWraper(chance=1.0, maxSigma=13)

    for i in range(len(imgs)):
        img = imgs[i]

        img = tr.resize(img, (480, 360))
        img2 = img.copy()
        img2 = bgwarper.run(img2)
        img2 = noiser.run(img2)

        tr.showImgs([img, img2])
    return


def testImgs():
    """test background wraper"""
    #adir = "../data/driver/"
    adir = "/Users/songbin/dev/data/docID/small/"
    imgs = loadImgs(adir)
    testEffect(imgs)
    #testPerformance(imgs)
    return


def main():
    #test()
    #testImgs()
    testBg()
    return


if __name__ == "__main__":
    utils.setupLog()
    main()
