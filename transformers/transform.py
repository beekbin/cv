from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import random
import cv2
import numpy as np
import numbers
import types
import collections
import warnings
import matplotlib.pyplot as plt
import logging

log = logging.getLogger("rotate-image")
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


# The input of the operations should be cv2 image
def loadImage(fname):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    # img.shape = (w, h, 3)
    log.debug(img.shape)
    return img


def showImg(img, title=None):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()    
    return


def showImgs(imgs, title="title"):
    windows = len(imgs)
    for i, img in enumerate(imgs):
        ax = plt.subplot(1, windows, i+1)
        ax.axis("off")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("%s-%s"%(title, i))
    plt.show()    
    return


def showTensorImg(ts, title):
    """show a CWH tensor in plt (WHW)"""
    img = np.transpose(ts, (1, 2, 0))
    showImg(img, title)
    return


def toTensor(img):
    """Convert a cv2 image: np.array(WxHxC) to caffe2 np.array(CxWxH)"""
    ts = np.transpose(img, (2, 0, 1))
    return ts


def addNoise(img, sigma=2.0, mean=0):
    """Add Gaussian Noise to the image"""
    img2 = np.random.normal(mean, sigma, size=img.shape)

    img2 += img
    img2 = np.clip(np.uint8(img2), 0, 255)
    return img2


## Section 2: adjust colors: birghtness, contrast, saturation, and hue
def adjustBrightness(img, fac):
    """fac should be [0.5, 1.1]"""
    img2 = np.float32(img) * fac
    img2 = img2.clip(min=0, max=255)
    return np.uint8(img2)


def adjustContrast(img, fac):
    """fac should be [0.5, 2.0]"""
    img2 = np.float32(img)
    mean = round(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).mean())
    img2 = (1-fac) * mean + fac * img2
    img2 = np.uint8(img2.clip(min=0, max=255))
    return img2


def adjustSaturation(img, fac):
    img2 = np.float32(img)
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

    img2 = (1-fac) * tmp + fac * img2
    img2 = np.uint8(img2.clip(min=0, max=255)) 
    return img2


def adjustHue(img, fac):
    """ fac should be in [-0.5, 0.5]
    NOTE: should not use this transformer.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(fac * 255)
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return img2


def adjustGamma(img, gamma, gain=1):
    img2 = np.float32(img)
    img2 = 255.0 * gain * np.power(img2/255.0, gamma)
    img2 = np.uint8(img2.clip(min=0, max=255))  
    return img2


def simpleRotate(img, angle, scale=0.85):
    """angle can be [0, 360].
    Filled with random color.
    TODO1: automatically calculate the scale needed;
    """
    fcolor = _genRandomColor()
    h, w, _ = img.shape
    center = (w/2, h/2)
    tmp = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(img, tmp, (w, h), borderValue=fcolor)
    return img2


def resize(img, size):
    return cv2.resize(img, dsize=size)


def paste(bg_img, img, x_offset, y_offset):
    """need to check boundries."""
    y1, y2 = y_offset, y_offset + img.shape[0]
    x1, x2 = x_offset, x_offset + img.shape[1]

    c = img.shape[2]
    try:
        bg_img[y1:y2, x1:x2, 0:c] = img[:, :, 0:c]
    except IndexError:
        log.error("index exception.")    
    return bg_img


def doAdjustAspectRatio(img, ratio):
    """
    The resulting image size will be different from the input img.
    """
    #1. resize img with-regard to the new ratio
    h, w, _ = img.shape
    r = float(w)/h
    if math.fabs(ratio - r) < 0.005:
        log.debug("abort adjust aspect ratio: %.3f Vs. %.3f" % (ratio, r))
        return None
    
    h2 = int(w / ratio)
    w2 = int(h * ratio)

    w_new = w
    h_new = h2
    if w2*h > w * h2:
        w_new = w2
        h_new = h

    scale = 1.0/max(w_new/w, h_new/h)
    inner = cv2.resize(img, dsize=(w_new, h_new))
    inner = cv2.resize(inner, None, fx=scale, fy=scale)
    log.debug("%.3f inner.shape=%s" % (scale, inner.shape))
    return inner


def adjustAspectRatio(img, ratio, bg_img=None):
    """ratio = width/height.
       Return a new image with the same size as the original one, gaussian filled, or bg_img.
    """
    inner = doAdjustAspectRatio(img, ratio)
    if inner is None:
        return img

    #2. paste the inner img to background image
    if bg_img is None:
        sigma = 5.0
        img2 = np.uint8(np.random.normal(0, sigma, size=img.shape))
    else:
        img2 = bg_img.copy()
    
    def _getRandom(n):
        if n < 1:
            return 0
        return random.randint(0, n)

    x_offset = _getRandom(img2.shape[1] - inner.shape[1])
    y_offset = _getRandom(img2.shape[0] - inner.shape[0])
    #x_offset, y_offset = 0, 
    #log.debug("(%d, %d) %s, %s" % (x_offset, y_offset, img2.shape, inner.shape))
    img2 = paste(img2, inner, x_offset, y_offset)
    return img2


def gradualShade(img, brightness, direction='lr'):
    h, w, _ = img.shape
    img2 = np.float32(img)

    if direction == "lr":
        alpha = -1.0 * brightness/5
        sign = -1
    elif direction == "rl":
        alpha = -4.0 * brightness/5
        sign = 1
        
    delta = brightness / float(w)

    for i in range(w):
        alpha += delta
        t = 1 + sign * alpha

        img2[:, i, :] = img2[:, i, :] * t
        #for j in range(h):
        #    for k in range(c):
        #        img2[j, i, k] = min(img2[j, i, k] * t, 255)

    img2 = np.uint8(img2.clip(0, 255))
    return img2


def crop(img, size, point=(0, 0)):
    """"""
    y, x = point
    w, h = size
    hf, wf, _ = img.shape

    if not isinstance(x, int):
        y = min(int(wf * y), wf)
        x = min(int(hf * x), hf)

    if not isinstance(w, int):
        w = int(wf * w)
        h = int(hf * h)

    x2 = min(x + h, hf) - 1
    y2 = min(y + w, wf) - 1
    log.info("w = %d, x2=%d, %s"%(w, x2, img.shape))
    img2 = img[x:x2, y:y2, :].copy()
    return img2


def rotate2(img):
    """TODO: change perspective"""
    h, w, _ = img.shape
    #pts1 = np.float32([[50,50],[200,50],[50,200]])
    #pts2 = np.float32([[10,100],[200,50],[100,250]])

    #pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    #pts2 = np.float32(([0+8, 0], [w, 10], [0, h-8], [w, h]))

    delta = int(0.2 * h)
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32(([delta, 0], [w, delta], [0, h-delta]))

    M = cv2.getAffineTransform(pts1, pts2)
    img2 = cv2.warpAffine(img,M,(w,h))
    return img2


def _genRandomColor():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r) 


def adjustPerspective(img, fac=0.15):
    """change perspective
    fac should be [0.05, 0.2]
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    TODO: a single parameter indicate the one of the 8 perspectives.
    """
    h, w, _ = img.shape
    
    dh = int(fac * w)
    dw = int(fac * h)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    views = []

    #1. from left to right
    pts2 = np.float32([[0, 0], [w-dw, dh], [0, h], [w-dw, h-dh]])
    views.append(pts2)

    #2. from right to left
    pts2 = np.float32([[dw, dh], [w, 0], [dw, h-dh], [w, h]]) 
    views.append(pts2)

    #3. from bottom to head
    pts2 = np.float32([[dw, dh], [w-dw, dh], [0, h], [w, h]])
    views.append(pts2)

    #4. from header to bottom
    pts2 = np.float32([[0, 0], [w, 0], [dw, h-dh], [w-dw, h-dh]])
    views.append(pts2)

    ##5. from top-left to bottom-right
    pts2 = np.float32([[0, 0], [w-dw/2, dh/2], [dw/2, h-dh/2], [w-dw, h-dh]])
    views.append(pts2) 
    #6. from bottom-right to top-left
    pts2 = np.float32([[dw, dh], [w-dw/2, dh/2], [dw/2, h-dh/2], [w, h]]) 
    views.append(pts2)
    pts2 = np.float32([[0, 0], [w-dw/2, dh/2], [dw/2, h-dh/2], [w, h]])
    views.append(pts2) 

    #7. from top-right to bottom-left
    pts2 = np.float32([[dw/2, dh/2], [w, 0], [dw, h-dh], [w-dw/2, h-dh/2]])
    views.append(pts2) 
    pts2 = np.float32([[dw/2, dh/2], [w, 0], [0, h], [w-dw/2, h-dh/2]]) 
    views.append(pts2)

    #8. from bottom-left to top-right
    pts2 = np.float32([[dw/2, dh/2], [w-dw, dh], [0, h], [w-dw/2, h-dh/2]])
    views.append(pts2)

    
    pts2 = views[4]
    fcolor = _genRandomColor()
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img2 = cv2.warpPerspective(img, M, (w, h), borderValue=fcolor)

    ##  get it back
    #M = cv2.getPerspectiveTransform(pts2, pts1)
    #img3 = cv2.warpPerspective(img2, M, (w, h))
    return img2


def testImgs():
    fnames = [
         "./data/driver/songbin.png",
         "./data/driver/songbin2.png",
        "./data/driver/motorist.jpg",
        "./data/driver/minnesota.png",
        "./data/driver/california.png",
        "./data/driver/ny2.png",
        "./data/driver/missi.png",
        "./data/driver/vermont.png",
        "./data/driver/florida.jpg",
        "./data/driver/ma.jpg",
    ]
    imgs = []
    for fname in fnames:
        img = loadImage(fname)
        imgs.append(img)

    tranImgs(imgs)   
    return


def tranImgs(imgs):
    for img in imgs:
        img = resize(img, (480, 360))
        #img2 = addNoise(img, sgma=5.0)
        #img2 = adjustBrightness(img, 0.8)
        #img2 = adjustContrast(img, 2.0)
        #img2 = adjustSaturation(img, 4.0)
        #img2 = adjustHue(img, 0.2)
        #img2 = adjustGamma(img, 1.5)
        
        #img2 = simpleRotate(img, -15)
        #img2 = gradualShade(img, 0.9, "rl")
        #img2 = crop(img, (1.0, 0.3), point=(0.0, 0.0))
        #img2 = adjustAspectRatio(img, 1.0)
        img2 = adjustPerspective(img)
        showImgs([img, img2])
    return



def test():
    #fname = "./data/driver/songbin2.png"
    fname = "./data/driver/california.png"
    img = loadImage(fname)
    #showImg(img, "Songbin")

    #img2 = addNoise(img)
    #showImg(img, "Noise")
    #img2 = adjustBrightness(img, 0.8)
    #img2 = adjustContrast(img, 2.0)
    #img2 = adjustSaturation(img, 4.0)
    #img2 = adjustHue(img, 0.2)
    #img2 = adjustGamma(img, 1.5)
    #img2 = simpleRotate(img, 8)
    img2 = gradualShade(img, 1.1)
    showImgs([img, img2])
    return


def main():
    #test()
    testImgs()
    return


if __name__ == "__main__":
    setupLog()
    main()