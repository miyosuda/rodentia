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
    
    target_joint_angles = np.array([0.0, 0.1, 0.2, 0.3,
                                    1.0, 0.9, 0.8, 0.7],
                                   dtype=np.float32)

    for i in range(3):
      obs = env.step(joint_angles=target_joint_angles)
    
    joint_angles = obs["joint_angles"]
    print(joint_angles)
    
    screen = obs["screen"]
    scipy.misc.imsave("../../debug.png", screen)

    # check shape
    self.assertEqual( (width,height,4), screen.shape )

    # dtype should be uint8
    self.assertEqual(np.uint8, screen.dtype)

if __name__ == '__main__':
  unittest.main()
