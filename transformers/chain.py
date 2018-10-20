from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import random
import math
import numpy as np
import transform as tr

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


class NoiseWraper:
    def __init__(self, chance, maxSigma=3):
        self.chance = chance
        self.maxSigma = maxSigma
        return
    
    def __str__(self):
        msg = "NoiseWraper: %.2f" % (self.chance)
        return msg 

    def run(self, img):
        if random.uniform(0, 1.0) > self.chance:
            return img

        mean = np.mean(img)
        sigma = min(0.05 * mean, self.maxSigma)  
        #sigma = random.uniform(1, self.maxSigma)
        return tr.addNoise(img, sigma)
    

class ColorWraper:
    def __init__(self, chance):
        self.bright_chance = chance
        self.contrast_chance = chance
        self.saturation_chance = chance
        self.gamma_chance = chance/2.0
        return

    def __str__(self):
        msg = "ColorWraper: %.2f" %(self.bright_chance)
        return msg    

    def run(self, img):
        img2 = img
        if random.uniform(0, 1.0) < self.bright_chance:
            fac = random.uniform(0.5, 1.1)
            img2 = tr.adjustBrightness(img2, fac)
            log.debug("adjust brightness: %.2f" % (fac))

        if random.uniform(0, 1.0) < self.contrast_chance:
            fac = random.uniform(0.5, 2.0)
            img2 = tr.adjustContrast(img2, fac)
            log.debug("adjust contrast: %.2f" % (fac))

        if random.uniform(0, 1.0) < self.saturation_chance:
            fac = random.uniform(0.4, 2.0)
            img2 = tr.adjustSaturation(img2, fac)
            log.debug("adjust saturation: %.2f" % (fac))

        if random.uniform(0, 1.0) < self.gamma_chance:
            gamma = random.uniform(0.6, 2.2)
            img2 = tr.adjustGamma(img2, gamma)
            log.debug("adjust gamma: %.2f" % (gamma))
        return img2


class AspectWraper:
    def __init__(self, chance):
        self.chance = chance
        self.fac_low = 0.7
        self.fac_high = 2.0
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
        self.fac_high = 1.1
        self.mean = (self.fac_low + self.fac_high)/2
        self.sigma = (self.mean - self.fac_low)**0.5
        return    

    def __str__(self):
        msg = "ShadowWraper: %.2f" % (self.chance)
        return msg

    def run(self, img):
        if random.random() > self.chance:
            return img

        fac = random.uniform(self.fac_low, self.fac_high)
        log.debug("shadow factor: %.3f" % (fac))

        choice = random.randint(1, 4)
        if choice == 1:
            img2 = tr.gradualShadeH(img, fac, 0)
        elif choice == 2:
            img2 = tr.gradualShadeH(img, fac, 1)
        elif choice == 4:
            img2 = tr.gradualShadeV(img, fac, 0)
        else:
            img2 = tr.gradualShadeV(img, fac, 1)

        return img2