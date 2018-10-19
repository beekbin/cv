from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import random
import math
import transform as tr

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


class NoiseWarper:
    def __init__(self, chance, maxMean=0, maxSigma=5):
        self.chance = chance
        self.maxSigma = maxSigma
        self.mean = maxMean
        return
    
    def __str__(self):
        msg = "NoiseWarper: %.2f" % (self.chance)
        return msg 

    def run(self, img):
        if random.uniform(0, 1.0) > self.chance:
            return img

        sigma = random.uniform(1, self.maxSigma)
        return tr.addNoise(img, sigma, self.mean)
    

class ColorWarper:
    def __init__(self, chance):
        self.bright_chance = chance
        self.contrast_chance = chance
        self.saturation_chance = chance
        self.gamma_chance = chance
        return

    def __str__(self):
        msg = "ColorWarper: %.2f" %(self.bright_chance)
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
            gamma = random.uniform(0.6, 2.4)
            img2 = tr.adjustGamma(img2, gamma)
            log.debug("adjust gamma: %.2f" % (gamma))
        return img2


class AspectWarper:
    def __init__(self, chance):
        self.chance = chance
        return

    def run(self, img):

        return