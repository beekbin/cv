from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import random
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


## References
## 1. Inspired by TorchVision
##    https://github.com/pytorch/vision/tree/master/torchvision/transforms
## 2. Implement based on Yu-Zhiyang's project
## https://github.com/YU-Zhiyang/opencv_transforms_torchvision/tree/master/cvtorchvision
## 3. OpenCV Vs. PIL: OpenCV2 is about 3+ times faster than PIL.
##    https://www.kaggle.com/vfdev5/pil-vs-opencv


## New
## 1. Remove torch dependence, so it can be used for Caffe2, pytorch,
##    or other DL frameworks;
## 2. Add Face detection;


## NOTE:
#  1. The input of the operations should be cv2 image (numpy.array);
#  2. The output of the operations should be cv2 image (numpy.array);


def loadImage(fname):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    if img is None:
        log.error("failed to load img: %s" % (fname))
        return None
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
        ax = plt.subplot(1, windows, i + 1)
        ax.axis("off")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("%s-%s" % (title, i))
    plt.show()
    return


def saveImgs(imgs, fname, title="title"):
    windows = len(imgs)
    for i, img in enumerate(imgs):
        ax = plt.subplot(1, windows, i + 1)
        ax.axis("off")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("%s-%s" % (title, i))
    plt.savefig(fname, bbox_inches='tight')
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
    img2 = np.uint8(img2.clip(0, 255))
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
    img2 = (1 - fac) * mean + fac * img2
    img2 = np.uint8(img2.clip(min=0, max=255))
    return img2


def adjustSaturation(img, fac):
    """fac should be [0, 2.0] """
    img2 = np.float32(img)
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

    img2 = (1 - fac) * tmp + fac * img2
    img2 = np.uint8(img2.clip(0, 255))
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
    """gamma should be [0.6, 3.0]:
        larger gamma --> darker img;
    """
    img2 = np.float32(img)
    img2 = 255.0 * gain * np.power(img2 / 255.0, gamma)
    img2 = np.uint8(img2.clip(min=0, max=255))
    return img2


def resize(img, size):
    h, w, _ = img.shape
    if (w, h) == size:
        return img
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


def randomPaste(bg_img, img):
    """randomly paste img into bg_img."""
    if img.shape[0] > bg_img.shape[0] or img.shape[1] > bg_img.shape[1]:
        log.error("Failed to paste: inner is bigger.")
        return img

    x_offset = random.randint(0, bg_img.shape[1] - img.shape[1])
    y_offset = random.randint(0, bg_img.shape[0] - img.shape[0])

    img2 = paste(bg_img, img, x_offset, y_offset)
    return img2


def doAdjustAspectRatio(img, ratio):
    """
    The resulting image size will be different from the input img.
    """
    #1. resize img with-regard to the new ratio
    h, w, _ = img.shape
    r = float(w) / h
    if math.fabs(ratio - r) < 0.005:
        log.debug("abort adjust aspect ratio: %.3f Vs. %.3f" % (ratio, r))
        return None

    h2 = int(w / ratio)
    w2 = int(h * ratio)

    w_new = w
    h_new = h2
    if w2 * h > w * h2:
        w_new = w2
        h_new = h

    scale = 1.0 / max(w_new / w, h_new / h)
    inner = cv2.resize(img, dsize=(w_new, h_new))
    inner = cv2.resize(inner, None, fx=scale, fy=scale)
    return inner


def adjustAspectRatio(img, ratio, bg_img=None):
    """ratio = width/height.
       Return a new image with the same size as the original one,
       gaussian filled, or bg_img.
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

    img2 = randomPaste(img2, inner)
    return img2


def gradualShadeV(img, brightness, direction=0, min_b=0.35, max_b=1.5):
    """add gradual shadow vertically.
       direction: 0: get darker from top to bottom;
                  1: get darker from bottom to top;
       minimum and maximum brightness:0.35, 1.5
    """
    h, _, _ = img.shape
    img2 = np.float32(img)

    half = brightness / 2.0
    alpha = max(min_b, 1 - half)
    beta = min(max_b, alpha + brightness)
    delta = (beta - alpha) / float(h)

    t = alpha
    if direction % 2 == 0:
        t = beta
        delta = -1 * delta

    for j in range(h):
        t += delta
        img2[j, :, :] = img2[j, :, :] * t

    img2 = np.uint8(img2.clip(0, 255))
    return img2


def gradualShadeH(img, brightness, direction=0, min_b=0.35, max_b=1.5):
    """
    add gradual shadow horizontally.
    direction: 0: get darker from left to right;
               1: get darker from right to left;
    minimum and maximum brightness:0.35, 1.5
    """
    _, w, _ = img.shape
    img2 = np.float32(img)

    min_b = 0.35
    max_b = 1.5

    half = brightness / 2.0
    alpha = max(min_b, 1 - half)
    beta = min(max_b, alpha + brightness)
    delta = (beta - alpha) / float(w)

    t = alpha
    if direction % 2 == 0:
        t = beta
        delta = -1 * delta

    for i in range(w):
        t += delta
        img2[:, i, :] = img2[:, i, :] * t

    img2 = np.uint8(img2.clip(0, 255))
    return img2


def gradualShade(img, fac, direction1=0, direction2=1):
    """
    fac: [0.5, 1.1]
    direction1: {0, 1}, 0: horizontally, 1: vertically;
    direction2: {0, 1}, 0: left->right/top-bottom, 1: right->left/bottom->top;
    """
    if direction1 % 2 == 0:
        img2 = gradualShadeH(img, fac, direction2)
    else:
        img2 = gradualShadeV(img, fac, direction2)

    return img2


def regionShadow(img, fac, region):
    """
    region: add glare/shadow in a region.
    """
    return


def crop(img, size, point=(0, 0)):
    """crop a rectangle region from point with size(w, h)."""
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
    log.debug("w = %d, x2=%d, %s" % (w, x2, img.shape))
    img2 = img[x:x2, y:y2, :].copy()
    return img2


def rotate2D(img, angle, scale=0.90):
    """
    angle: [-180, 180];
    scale: [0, 1.0];
    Filled with random color.

    NOTE: if rotate degree less than 90, use rotateX instead.

    TODO1: automatically calculate the scale needed;
    """
    if math.fabs(math.tan(angle * math.pi / 180.0)) > 0.8:
        if scale > 0.80:
            scale = 0.80

    h, w, _ = img.shape
    center = (w / 2.0, h / 2.0)
    tmp = cv2.getRotationMatrix2D(center, angle, scale)

    fcolor = _genRandomColor()
    img2 = cv2.warpAffine(img, tmp, (w, h), borderValue=fcolor)
    return img2


def rotate2DBound(image, angle):
    """make sure the whole image remain in the bound"""
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    img2 = cv2.warpAffine(image, M, (nW, nH))
    img2 = cv2.resize(img2, (w, h))
    return img2

def rotate2DX(img, fac):
    """
    approximatively rotate the image [-90, 90] degrees.
    This operation can make sure the image is inside the box.
    fac: [-0.99, 0.99]
    """
    if math.fabs(fac) > 0.99:
        log.warn("fac should be [-0.99, 0.99]")
        return img

    h, w, _ = img.shape
    dw = int(fac * h)
    dh = int(fac * w)

    ## needs to pick three points
    if fac > 0:
        dh = min(dh, h)
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[dw, 0], [w, dh], [0, h - dh]])
    else:
        dw, dh = -dw, -dh
        dh = min(dw, h)
        pts1 = np.float32([[0, 0], [0, h], [w, h]])
        pts2 = np.float32([[0, dh], [dw, h], [w, h - dh]])

    M = cv2.getAffineTransform(pts1, pts2)
    fcolor = _genRandomColor()
    img2 = cv2.warpAffine(img, M, (w, h), borderValue=fcolor)
    return img2


def _genRandomColor():
    """gen a random BGR color."""
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r)


def _genRandomImg(shape, sigma=5.0):
    if random.random() > 0.5:
        bg_img = np.uint8(np.random.normal(0, sigma, size=shape))
    else:
        bg_img = np.zeros(shape, dtype=np.uint8)
        for c in range(shape[2]):
            bg_img[:, :, c] = random.randint(0, 255)

    assert bg_img.shape == shape, "unexpected shape."        
    return bg_img


def adjustPerspectiveX(img, idx=-1, fac=0.15, scale=(1.0, 1.0)):
    """approximatively perspective change.
    fac should be [0.05, 0.2]
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
       py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    TODO:
      1. use switch-case to compute only the needed pts2;
      2. adjust the corner with scale taking into account;
    """
    h1, w1, _ = img.shape

    w, h = int(w1 * scale[0]), int(h1 * scale[1])
    aw = (w1 - w) // 2
    ah = (h1 - h) // 2

    dh = int(fac * w)
    dw = int(fac * h)
    pts1 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])

    views = []
    #1. from left to right
    #pts2 = np.float32([[0, 0], [w-dw, dh], [0, h], [w-dw, h-dh]])
    pts2 = np.float32([[aw, ah], [w - dw, dh], [aw, h - ah], [w - dw, h - dh]])
    views.append(pts2)

    #2. from right to left
    pts2 = np.float32([[dw, dh], [w, 0], [dw, h - dh], [w, h]])
    views.append(pts2)

    #3. from bottom to head
    pts2 = np.float32([[dw, dh], [w - dw, dh], [0, h], [w, h]])
    views.append(pts2)

    #4. from header to bottom
    pts2 = np.float32([[0, 0], [w, 0], [dw, h - dh], [w - dw, h - dh]])
    views.append(pts2)

    ##5. from top-left to bottom-right
    pts2 = np.float32([[0, 0], [w - dw/2, dh/2], [dw/2, h-dh/2], [w-dw, h-dh]])
    views.append(pts2)

    #6. from bottom-right to top-left
    pts2 = np.float32([[dw, dh], [w-dw/2, dh/2], [dw/2, h-dh/2], [w, h]])
    views.append(pts2)
    pts2 = np.float32([[0, 0], [w-dw/2, dh/2], [dw/2, h-dh/2], [w, h]])
    views.append(pts2)

    #7. from top-right to bottom-left
    pts2 = np.float32([[dw/2, dh/2], [w, 0], [dw, h-dh], [w-dw/2, h-dh/2]])
    views.append(pts2)

    #8. from bottom-left to top-right
    pts2 = np.float32([[dw/2, dh/2], [w-dw, dh], [0, h], [w-dw/2, h-dh/2]])
    views.append(pts2)
    pts2 = np.float32([[dw/2, dh/2], [w, 0], [0, h], [w-dw/2, h-dh/2]])
    views.append(pts2)

    if idx < 0:
        idx = random.randint(0, len(views) - 1)
    else:
        idx = idx % len(views)

    pts2 = views[idx]
    fcolor = _genRandomColor()
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img2 = cv2.warpPerspective(img, M, (w, h),
               borderMode=cv2.BORDER_CONSTANT, borderValue=fcolor)

    ##  get it back
    #M = cv2.getPerspectiveTransform(pts2, pts1)
    #img3 = cv2.warpPerspective(img2, M, (w, h))

    if w != w1 or h != h1:
        bg_img = _genRandomImg(img.shape)
        img2 = randomPaste(bg_img, img2)
    
    return img2


def adjustPerspective(img, anglex=0, angley=0, anglez=0, shear=0, fov=45,
                translate=(0, 0), scale=(0.85, 0.85),
                resample=cv2.INTER_LINEAR, fillcolor=None):
    """
    Precisely perspective change.
    Note: No need simpleRotate after this operation.

    This function is from YU-Zhiyang:
 https://github.com/YU-Zhiyang/opencv_transforms_torchvision/tree/master/cvtorchvision

    anglex, angley, anglez, shear: [-180, 180]
    """

    imgtype = img.dtype
    h, w, _ = img.shape
    centery = h * 0.5
    centerx = w * 0.5

    alpha = math.radians(shear)
    beta = math.radians(anglez)

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) - sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) + cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12], [0, 0, 1]],
                        dtype=np.float32)
    # -------------------------------------------------------------------------------
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(math.radians(fov / 2))

    radx = math.radians(anglex)
    rady = math.radians(angley)

    sinx = math.sin(radx)
    cosx = math.cos(radx)
    siny = math.sin(rady)
    cosy = math.cos(rady)

    r = np.array([[cosy, 0, -siny, 0],
                  [-siny * sinx, cosx, -sinx * cosy, 0],
                  [cosx * siny, sinx, cosx * cosy, 0],
                  [0, 0, 0, 1]])

    pcenter = np.array([centerx, centery, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    perspective_matrix = cv2.getPerspectiveTransform(org, dst)
    #total_matrix = perspective_matrix @ affine_matrix
    total_matrix = np.matmul(perspective_matrix, affine_matrix)

    if fillcolor is None:
        fillcolor = _genRandomColor()
    result_img = cv2.warpPerspective(img, total_matrix, (w, h), flags=resample,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    return result_img.astype(imgtype)


def _randomFill(img, area, mean, sigma):
    h0, w0, _ = img.shape
    x, y, w, h = area

    #1. expand it to cover hair and neck.
    dw = int(w * 0.1)
    dh = int(h * 0.15)

    if x > dw:
        x -= dw
        w += dw

    if y > dh:
        y -= dh
        h += dh

    if x + w + dw < w0:
        w += dw

    if y + h + dh < h0:
        h += dh

    #2. erase the face area
    rnd = np.random.normal(mean, sigma, (h, w, 3))
    try:
        img[y:y + h, x:x + w] = np.uint8(rnd)
    except IndexError as e:
        log.error("Index error: %s" % (e))
        log.error("area=%s, shape=%s" % (area, img.shape))
    except ValueError as e:
        log.error("Value error: %s" % (e))
        log.error("area=%s, shape=%s" % (area, img.shape))
    return


def eraseFace(model, img, mean=127, sigma=8):
    """Use CascadeClassifier to detect faces,
       and erase the area with random filling.

       model: is a cv2.CascadeClassifier;
       model file can be found:
          https://github.com/opencv/opencv/tree/master/data/haarcascades
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.3, 5)

    img2 = img
    num = min(2, len(faces))  # detect at most 2 faces
    for i in range(num):
        #x, y, w, h = faces[i]
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0,255), 3)
        # expand it to cover hair and neck.
        #area = (x, y, w, h)
        _randomFill(img2, faces[i], mean, sigma)

    return img2
