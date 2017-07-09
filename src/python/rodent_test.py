# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import scipy.misc

import rodent

class RodentTest(unittest.TestCase):
  def testVersion(self):
    version = rodent.version();
    self.assertEqual(version, "0.1")

  def testEnv(self):
    width  = 84 * 4
    height = 84 * 4
    env = rodent.Env(width=width, height=height)
    
    action = np.array([10, 0, 0], dtype=np.int32)

    for i in range(3):
      obs = env.step(action=action)
    
    screen = obs["screen"]
    scipy.misc.imsave("debug.png", screen)

    # check shape
    self.assertEqual( (width,height,4), screen.shape )

    # dtype should be uint8
    self.assertEqual(np.uint8, screen.dtype)

if __name__ == '__main__':
  unittest.main()
