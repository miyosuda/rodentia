# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np

import rodent

class RodentTest(unittest.TestCase):
  def testVersion(self):
    version = rodent.version();
    self.assertEqual(version, "0.1")

  def testEnv(self):
    env = rodent.Env()
    
    joint_angles = np.array([1.0, 2.0, 3.0, 4.0,
                             10.0, 102.0, 103.0, 104.0],
                            dtype=np.float32)
    
    ret = env.step(joint_angles=joint_angles)
    print(ret)

if __name__ == '__main__':
  unittest.main()
