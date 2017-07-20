# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import scipy.misc

import rodent_module

def to_nd_float_array(list_obj):
  return np.array(list_obj, dtype=np.float32)
  

class RodentTest(unittest.TestCase):
  def testVersion(self):
    version = rodent_module.version();
    self.assertEqual(version, "0.0.1")

  def testEnv(self):
    width  = 84 * 4
    height = 84 * 4
    env = rodent_module.Env(width=width, height=height)

    # Check setup interfaces
    # Add box
    env.add_box(half_extent=to_nd_float_array([5.0, 5.0, 5.0]),
                pos=to_nd_float_array([10.0, 5.0, 10.0]),
                rot=0.0,
                detect_collision=False)

    # Add Sphere
    sphere_id = env.add_sphere(radius=2.0,
                               pos=to_nd_float_array([-5.0, 2.0, -5.0]),
                               rot=0.0,
                               detect_collision=True)
    print("sphere_id={}".format(sphere_id))

    # Locate agent
    env.locate_agent(pos=to_nd_float_array([0.0, 1.0, 0.0]),
                     rot=0.0)

    # Check step with action
    action = np.array([10, 0, 0], dtype=np.int32)

    for i in range(3):
      obs = env.step(action=action)
    
    screen = obs["screen"]
    scipy.misc.imsave("debug.png", screen)

    # Check shape
    self.assertEqual( (width,height,4), screen.shape )

    # dtype should be uint8
    self.assertEqual(np.uint8, screen.dtype)

    collided = obs["collided"]

    print("collided size={}".format(len(collided)))

if __name__ == '__main__':
  unittest.main()
