from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import random
import math
import string
import numpy as np
import cv2
import os

import transform as tr
import utils

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


class ResizeWraper:
    """resize the image to desired size"""
    def __init__(self, w, h, chance=1.0):
        self.chance = chance
        self.w = w
        self.h = h
    
    def __str__(self):
        msg = "ResizeWraper: %.2f" % (self.chance)
        return msg

    def run(self, img):
        img2 = tr.resize(img, (self.w, self.h))
        return img2


class NoiseWraper:
    def __init__(self, chance=0.5, maxSigma=3):
        self.chance = chance
        self.maxSigma = maxSigma
        self.meanRange = (-3, 3)
        return

    def __str__(self):
        msg = "NoiseWraper: %.2f" % (self.chance)
        return msg

    def setMeanRange(self, range):
        if len(range) != 2:
            log.error("range should be [a, b]")
            return
        self.meanRange = range
        return

    def run(self, img):
        if random.uniform(0, 1.0) > self.chance:
            return img

        mean = random.uniform(self.meanRange[0], self.meanRange[1])
        sigma = random.uniform(self.maxSigma/2.0, self.maxSigma)

        #amean = np.mean(img)
        #log.info("amean: %.3f" % (amean))
        #sigma = min(0.05 * amean, self.maxSigma)  
        return tr.addNoise(img, sigma, mean)


class ColorWraper:
    def __init__(self, chance=0.5, gamma_chance=0.2):
        self.bright_chance = chance
        self.contrast_chance = chance
        self.saturation_chance = chance
        self.gamma_chance = gamma_chance
        return

    def __str__(self):
        msg = "ColorWraper: %.2f" %(self.bright_chance)
        return msg    

    def run(self, img):
        img2 = img
        if random.uniform(0, 1.0) < self.bright_chance:
            fac = random.uniform(0.6, 1.1)
            img2 = tr.adjustBrightness(img2, fac)
            log.debug("adjust brightness: %.2f" % (fac))

        if random.uniform(0, 1.0) < self.contrast_chance:
            fac = random.uniform(0.6, 2.0)
            img2 = tr.adjustContrast(img2, fac)
            log.debug("adjust contrast: %.2f" % (fac))

        if random.uniform(0, 1.0) < self.saturation_chance:
            fac = random.uniform(0.5, 2.0)
            img2 = tr.adjustSaturation(img2, fac)
            log.debug("adjust saturation: %.2f" % (fac))

        if random.uniform(0, 1.0) < self.gamma_chance:
            gamma = random.uniform(0.6, 2.2)
            img2 = tr.adjustGamma(img2, gamma)
            log.debug("adjust gamma: %.2f" % (gamma))
        return img2


class AspectWraper:
    def __init__(self, chance=0.5):
        self.chance = chance
        self.fac_low = 0.8
        self.fac_high = 1.5
        return

    def __str__(self):
        msg = "AspectWraper: %s" % (self.chance)
        return msg

    def run(self, img):
        if random.random() > self.chance:
            return img
        
        fac = random.uniform(self.fac_low, self.fac_high)
        h, w, _ = img.shape
        ratio = (fac * w) / h
        img2 = tr.adjustAspectRatio(img, ratio)
        return img2


class ShadowWraper:
    """NOTE: ShadowWraper should not use with ColorWraper."""
    def __init__(self, chance=0.5):
        self.chance = chance
        self.fac_low = 0.5
        self.fac_high = 1.5
        self.min_brightness = 0.35
        self.max_brightness = 1.5
        self.mean = (self.fac_low + self.fac_high) / 2
        self.sigma = (self.mean - self.fac_low)**0.5
        return

    def __str__(self):
        msg = "ShadowWraper: %.2f" % (self.chance)
        return msg

    def setBrightnessRange(self, min_b, max_b):
        self.min_brightness = min_b
        self.max_brightness = max_b
        return

    def run(self, img):
        if random.random() > self.chance:
            return img

        fac = random.uniform(self.fac_low, self.fac_high)
        log.debug("shadow factor: %.3f" % (fac))

        a, b = self.min_brightness, self.max_brightness
        choice = random.randint(1, 4)
        if choice == 1:
            img2 = tr.gradualShadeH(img, fac, 0, a, b)
        elif choice == 2:
            img2 = tr.gradualShadeH(img, fac, 1, a, b)
        elif choice == 3:
            img2 = tr.gradualShadeV(img, fac, 0, a, b)
        else:
            img2 = tr.gradualShadeV(img, fac, 1, a, b)

        return img2



class ShrinkWraper:
    """Rescale the ID with regard to the whole image.
       Will keep the input aspect ratio.
    """
    def __init__(self, chance=0.5, fac_low=0.65, bg_imgs=None):
        self.chance = chance
        self.fac_low = fac_low
        self.fac_high = 1.0

        self.bg_imgs = bg_imgs
        if bg_imgs is None:
            self.bg_imgs = []
        return

    def __str__(self):
        msg = "ShrinkWraper: %.3f" % (self.chance)    
        return msg

    def resizeBbImgs(self, w, h):
        sz = (w, h)
        for i in range(len(self.bg_imgs)):
            img = self.bg_imgs[i]
            self.bg_imgs[i] = tr.resize(img, sz)
        return

    def _randomImg(self, size):
        """get a backgroud image randomly"""
        n = len(self.bg_imgs)
        index = random.randint(0, n + 2)
        if index < n:
            bg_img = self.bg_imgs[index].copy()
        elif index == n:
            # get a constatn color bg image
            bg_img = np.full(size, 0, np.uint8)
            fcolor = tr._genRandomColor()
            _, _, c = size
            for i in range(c):
                bg_img[:, :, i] = fcolor[i]
        else:
            # get a totally random bg image
            sigma = 4.0
            bg_img = np.random.normal(0.0, sigma, size)
            bg_img = np.uint8(bg_img)
        return bg_img

    def run(self, img):
        if random.random() > self.chance:
            return img

        #1. shrink it
        fac = random.uniform(self.fac_low, self.fac_high)
        log.debug("fac = %.3f" % (fac))
        if math.fabs(fac - 1.0) < 0.02:
            return img

        h, w, _ = img.shape
        h, w = int(fac*h), int(fac * w)
        img2 = tr.resize(img, (w, h))
        
        #2. past it to background image
        bg_img = self._randomImg(img.shape)

        #3. paste it to the background image
        dh = img.shape[0] - h
        dw = img.shape[1] - w
        x_offset = random.randint(0, dw)
        y_offset = random.randint(0, dh)

        img2 = tr.paste(bg_img, img2, x_offset, y_offset)
        return img2


class FaceWraper:
    """detect and erase faces.
    Using OpenCV Haar Feature-based Cascade Classifiers.
    https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
    model file:
    https://github.com/opencv/opencv/tree/master/data/haarcascades

    NOTE 1: face detecter may fail if the image is skewed. So should apply
    FaceWraper before resize/aspectRatio/rotate/noise.

    NOTE 2: face detecter is heavy, so try to avoid use it on the fly;

    """
    def __init__(self, fmodel, chance=0.5):
        self.chance = chance
        self.fmodel = fmodel
        self.model = cv2.CascadeClassifier(fmodel)
        return

    def __str__(self):
        msg = "FaceWraper: %.2f;\nmodel: %s" % (self.chance, self.fmodel)
        return msg

    def run(self, img):
        mean = random.uniform(0, 64)
        sigma = random.uniform(5, 15)
        img2 = tr.eraseFace(self.model, img, mean, sigma)
        return img2


class Rotate2DWraper:
    """Rotate image in 2D  around center.
    Note: cannot
    """
    def __init__(self, chance=0.5, scale=(0.80, 1.0), angle=(-180, 180)):
        self.chance = chance
        self.scale_low = scale[0]
        self.scale_high = scale[1]

        self.angle_low = angle[0]
        self.angle_high = angle[1]
        return

    def __str__(self):
        msg = "Rotate2DWraper: %.2f" % (self.chance)
        return msg

    def run(self, img):
        if random.random() > self.chance:
            return img
        
        scale = random.uniform(self.scale_low, self.scale_high)
        angle = random.uniform(self.angle_low, self.angle_high)

        img2 = tr.rotate2D(img, angle, scale)
        return img2


class Rotate2DXWraper:
    """Rotate image in 2D  around center.
    Approximatively rotate the image degrees.
    Note: the degree range is [-90, 90].
    """
    def __init__(self, chance=0.5, angle=(-0.99, 0.99)):
        self.chance = chance
        
        self.angle_low = angle[0]
        self.angle_high = angle[1]
        return

    def __str__(self):
        msg = "Rotate2DWraper: %.2f" % (self.chance)
        return msg

    def run(self, img):
        if random.random() > self.chance:
            return img
        
        fac = random.uniform(self.angle_low, self.angle_high)
        img2 = tr.rotate2DX(img, fac)
        return img2


class Rotate3DWraper:
    """Rotate image in 3D around center.
    Note: 
      (1) don't use it with Rotate2D(X); already rotate by Z;
      (2) don't use it with ShrinkWraper: already rescaled;
      (3) don't use it with AspectWraper: already implemented by different scales.
    """        
    def __init__(self, chance=0.5, x=(-40, 40), y=(-40, 40), z=(-40, 40)):
        self.chance = chance
        self.angle_x = x
        self.angle_y = y
        self.angle_z = z
        self.shear = (-180, 180)
        self.scale_range = (0.7, 1.0)
        return

    def __str__(self):
        msg = "Rotate3DXWraper: %.2f" % (self.chance)
        return msg

    def _adjustScale(self, z, scale_w, scale_h):
        """if angle_z is larger than 15 degrees, img should be shrinked a lot to avoid cut out"""
        if math.tan(45*math.pi/180) > 0.3:
            # 0.3 == math.tan(15 degree)
            if scale_w <= 0.85 and scale_h <= 0.85:
                return scale_w, scale_h

        r = scale_w / scale_h
        if scale_w > scale_h:
            scale_w = 0.85
            scale_h = scale_w / r
        else:
            scale_h = 0.85
            scale_w = scale_h * r
        return scale_w, scale_h

    def run(self, img):
        if random.random() > self.chance:
            return img
        

        x = random.uniform(self.angle_x[0], self.angle_x[1])
        y = random.uniform(self.angle_y[0], self.angle_y[1])
        z = random.uniform(self.angle_z[0], self.angle_z[1])
        shear = random.uniform(self.shear[0], self.shear[1])

        scale_w = random.uniform(self.scale_range[0], self.scale_range[1])
        if random.random() < 0.7:
            scale_h = scale_w
        else:
            scale_h = random.uniform(self.scale_range[0], self.scale_range[1])

        scale = self._adjustScale(z, scale_w, scale_h)
        img2 = tr.adjustPerspective(img, anglex=x, angley=y, anglez=z, shear=shear, scale=scale)
        return img2


class Rotate3DXWraper:
    """approximatively perspective change.
    """
    def __init__(self, chance=0.5, fac=(0.05, 0.2)):
        self.chance = chance 
        self.fac = fac
        self.scale_range = (0.8, 1.0)
        return

    def __str__(self):
        msg = "Rotate3DXWraper: %.2f" % (self.chance)
        return msg

    def run(self, img):
        if random.random() > self.chance:
            return img

        fac = random.uniform(self.fac[0], self.fac[1])
        fac = 0.2

        scale_w = random.uniform(self.scale_range[0], self.scale_range[1])
        scale_h = random.uniform(self.scale_range[0], self.scale_range[1])
        scale = (scale_w, scale_h)
        img2 = tr.adjustPerspectiveX(img, fac=fac, scale=scale)
        return img2 


class EraseWraper:
    """Randomly crop the upper header of each image.
    Note: don't use it with EraseWraper.
    """
    def __init__(self, chance=0.5):
        self.chance = chance
        self.mean = (0, 10)
        self.sigma = (2, 8)
        return    

    def __str__(self):
        msg = "EraseWraper: %.2f " % (self.chance)
        return msg

    def run(self, img):
        if random.random() > self.chance:
            return
        
        h, w, _ = img.shape

        x1 = random.randint(int(0.1*w), int(0.3*w))
        y1 = random.randint(int(0.33*h), int(0.5*h))

        x2 = random.randint(int(0.7*w), int(0.90*w))
        y2 = random.randint(int(0.55*h), int(0.90*h))

        mean = random.uniform(self.mean[0], self.mean[1])
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img[y1:y2, x1:x2] = np.uint8(np.random.normal(mean, sigma, (y2-y1, x2-x1, 3)))
        return img
    

class Chain:
    """A serial of Transformers"""
    def __init__(self, name):
        self.name = name
        self.operators = []
        return

    def __str__(self):
        msg = "%s has %d operators:" % (self.name, len(self.operators))
        for op in self.operators:
            msg = "%s\n\t%s" % (msg, op)

        return msg

    def addOperator(self, op):
        self.operators.append(op)
        return

    def run(self, img):
        if len(self.operators) < 1:
            return img

        img2 = img.copy()
        for op in self.operators:
            img2 = op.run(img2)
        return img2


class RandomObjects:
    """Add randomly objects to img.
    Objects: lines, text, eclipse.
    """
    def __init__(self):
        self.chars = string.ascii_uppercase + string.digits
        return

    def drawVerticalLine(self, img, num):
        h, w, _ = img.shape
        for _ in range(num):
            x1 = random.randint(5, w-5)
            x2 = random.randint(5, w-5)
            rcolor = tr._genRandomColor()
            thick = random.randint(1, 15)
            cv2.line(img, (x1, 0), (x2, h), rcolor, thick)
        return

    def drawHorizontalLine(self, img, num):
        h, w, _ = img.shape
        for _ in range(num):
            y1 = random.randint(5, h-5)
            y2 = random.randint(5, h-5)
            rcolor = tr._genRandomColor()
            thick = random.randint(1, 15)
            cv2.line(img, (0, y1), (w, y2), rcolor, thick) 
        return

    def drawLines(self, img):
        num = random.randint(2, 4)
        self.drawVerticalLine(img, num)

        num = random.randint(2, 4)
        self.drawHorizontalLine(img, num)
        return

    def drawSimpleLines(self, img, num):
        if random.random() < 0.5:
            self.drawVerticalLine(img, num)
        else:
            self.drawHorizontalLine(img, num)
        return

    def randomString(self, size):
        txt = ''.join(random.choice(self.chars) for _ in range(size))
        if random.random() < 0.5:
            txt = txt.lower()
        return txt

    def randomEdgePoint(self, w, h):
        if random.random() < 0.5:
            x = random.randint(10, w // 4)
        else:
            x = random.randint(2*w//3, w-5)

        if random.random() < 0.5:
            y = random.randint(10, h // 4)
        else:
            y = random.randint(2*h//3, h-5)

        return (x, y)

    def drawText(self, img):
        h, w, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        num = random.randint(2, 10)
        for _ in range(num):
            txt = self.randomString(num+6)
            rcolor = tr._genRandomColor()
            point = self.randomEdgePoint(w, h)
            scale = 2 * random.random() + 0.5
            thickness = random.randint(1, 5)
            cv2.putText(img, txt, point, font, scale, rcolor, thickness)
        return

    def drawEclipse(self, img):
        h, w, _ = img.shape

        num = random.randint(4, 10)
        for _ in range(num):
            pt1 = self.randomEdgePoint(w, h)
            x, y = pt1
            maxw = min(x, w-x)
            maxh = min(y, h-y)

            len1 = int(random.uniform(10, maxw))
            len2 = int(random.uniform(10, maxh))
            size = (len1, len2)

            angle = random.randint(-45, 45)
            thickness = random.randint(1, 10)
            if random.random() < 0.5:
                thickness = 0 - thickness

            color = tr._genRandomColor()
            cv2.ellipse(img, pt1, size, angle, 0, 360, color, thickness)
        return


class BackGroundWraper:
    """
    Add some randomly background to the image.
    Note: will randomly shrink the image as well.
    """
    def __init__(self, chance=0.5, fac_low=0.65):
        self.chance = chance
        self.bg_imgs = []
        self.fac_low = fac_low
        self.fac_high = 1.0
        self.objectDrawer = RandomObjects()
        return

    def reSizeBgImgs(self, w, h):
        for i in range(len(self.bg_imgs)):
            self.bg_imgs[i] = cv2.resize(self.bg_imgs[i], (w, h))
        return

    def loadImgs(self, fdir, size=None):
        fnames = utils.getFileNames(fdir)
        if len(fnames) < 1:
            log.error("Failed to load bg-images.")
            return

        for fname in fnames:
            img = tr.loadImage(fname)
            if img is None:
                log.warning("failed to load img: %s" % (fname))
                continue
            if size is not None:
                img = cv2.resize(img, size)
            self.bg_imgs.append(img)

        log.debug("load %d bg-images from %s" % (len(self.bg_imgs), fdir))    
        return

    def setImgs(self, imgs):
        self.bg_imgs = imgs
        return

    def addObjects(self, img):
        self.objectDrawer.drawLines(img)
        self.objectDrawer.drawText(img)
        self.objectDrawer.drawEclipse(img)
        return

    def _genRandomImg(self, size):
        bg_img = np.full(size, 0, np.uint8)
        fcolor = tr._genRandomColor()
        
        for i in range(size[2]):
            bg_img[:, :, i] = fcolor[i]
        return bg_img

    def _randomImg(self, shape):
        """get a backgroud image randomly"""
        n = len(self.bg_imgs)
        index = random.randint(0, 2*n + 2)
        if index < n:
            size = (shape[1], shape[0])
            bg_img = cv2.resize(self.bg_imgs[index], size)
        elif index % 2 == 0:
            # get a constatn color bg image
            bg_img = self._genRandomImg(shape)
        else:
            # get a totally random bg image
            sigma = 40.0
            bg_img = np.random.normal(128.0, sigma, shape)
            bg_img = np.uint8(bg_img)

        self.addObjects(bg_img)    
        return bg_img


    def run(self, img):
        if random.random() > self.chance:
            return img

        #1. shrink it
        fac = random.uniform(self.fac_low, self.fac_high)
        if math.fabs(fac - 1.0) < 0.02:
            return img

        h, w, _ = img.shape
        h, w = int(fac*h), int(fac * w)
        img2 = tr.resize(img, (w, h))

        #2. past it to background image
        bg_img = self._randomImg(img.shape)

        #3. paste it to the background image
        dh = img.shape[0] - h
        dw = img.shape[1] - w
        x_offset = random.randint(0, dw)
        y_offset = random.randint(0, dh)

        img2 = tr.paste(bg_img, img2, x_offset, y_offset)
        return img2

def genFlare(width, height, ring_amount=0):
    """Generate a image of one glare.
    the size of the image is (width, height, 3);
    """
    radius = np.random.uniform(height/4.0, height/2.0)
    base_amount = np.random.uniform(2.0, 4.0)
    color_b, color_g, color_r = np.random.uniform(0.7, 1.0, 3)
    centre_x, centre_y = np.random.uniform(0.3, 0.7, 2)
    icentre_x = centre_x * width
    icentre_y = centre_y * height

    mix = np.random.uniform(0.3, 0.7)

    log.debug("Radius: {}".format(radius))
    log.debug("Base amount: {}".format(base_amount))
    log.debug("Color: [{}, {}, {}]".format(color_b, color_g, color_r))
    log.debug("Centre: [{}, {}]".format(centre_x, centre_y))

    # rays = 200
    # ring_amount = 0.25
    #ring_amount = 0
    ray_amount = 0.15
    ring_width = 1.5
    linear = 0.025
    gauss = 0.005
    # falloff = 5.0
    # sigma = radius / 6
    pi = 3.1415926

    flare = np.zeros((height, width, 3), np.uint8)

    for row in range(height):
        for col in range(width):
            dx = col - icentre_x
            dy = row - icentre_y
            distance = math.sqrt(dx * dx + dy * dy)

            a = math.exp(-distance * distance * gauss) * mix + math.exp(-distance * linear) * (1 - mix)
            a *= base_amount

            # if distance > radius + ring_width:
            #     a = (distance - (radius + ring_width)) / falloff * 0.5 + a * 0.5

            if ring_amount > 0.01:
                if distance > (radius - ring_width) and distance < (radius + ring_width):
                    ring = abs(distance - radius) / ring_width
                    ring = 1 - ring * ring * (3 - 2 * ring)
                    ring *= ring_amount
                    a += ring

            angle = math.atan2(dx, dy) + pi
            # angle = (mod(angle / pi * 17 + 1.0 + Noise1(angle * 10, p, g1), 1.0) - 0.5) * 2
            angle = (((angle / pi * 17 + 1.0) % 1.0) - 0.5) * 2
            angle = abs(angle)
            angle **= 5

            b = ray_amount * angle / (1 + distance * 0.1)
            a += b

            a = min(max(0, a), 1) * 255

            flare[row, col] = a * color_b, a * color_g, a * color_r
    return flare


class GlareWraper:
    """
    Randomly add glares to images.
    """
    def __init__(self, chance=0.5):
        self.chance = chance

        # flares are list of all kinds of flares:
        #  (1) download from web
        #  (2) generated by genGlare() function
        self.flares = []

        # maximum number of flares and shadows
        self.num1 = 2
        self.num2 = 1
        self.glare_size = (0.15, 0.3)
        return

    def __str__(self):
        msg = "GlareWraper: %.3f" % (self.chance)
        return msg

    def genFlareTemplates(self, num=30, width=150, height=150):
        for _ in range(num):
            flare = genFlare(width, height)
            self.flares.append(flare)

        log.info("%d flares" % (len(self.flares)))
        return

    def addFlareTemplates(self, imgs):
        """
        add a list of imgs of flares.
        These imgs can be downlowd from web.
        """
        self.flares.extend(imgs)
        log.info("%d flares" % (len(self.flares)))
        return

    def addGlare(self, img, flag=True):
        """if flag == True, then will add a glare;
           else, will add a shadow.
        """
        flare = random.choice(self.flares)
        h, w = img.shape[0:2]

        low, high = self.glare_size
        gh = random.randint(int(low*h), int(high*h))
        gw = random.randint(int(low*w), int(high*w))
        #gh = int(0.2*h)
        #gw = int(0.2*w)

        x = random.randint(0, w - gw)
        y = random.randint(0, h - gh)

        flare = cv2.resize(flare, (gw, gh))

        z = img[y:y+gh, x:x+gw, :]
        if flag:
            #weight = 0.9
            weight = random.randint(60, 100) / 100.0
            z = z * weight + flare
        else:
            #weight = 0.7
            weight = random.randint(70, 100) / 100.0
            z = z * weight - flare

        z = np.clip(z, 0, 255)
        img[y:y+gh, x:x+gw, :] = z
        return

    def run(self, img):
        if random.random() > self.chance:
            return img

        for _ in range(self.num1):
            self.addGlare(img, True)

        for _ in range(self.num2):
            self.addGlare(img, False)
        return