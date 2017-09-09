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
  

class RodentModuleTest(unittest.TestCase):
  def testVersion(self):
    version = rodent_module.version();
    self.assertEqual(version, "0.0.1")

  def testEnv(self):
    width  = 84 * 4
    height = 84 * 4
    env = rodent_module.Env(width=width, height=height,
                            floor_size=to_nd_float_array([10,10]),
                            floor_texture_path="")

    # Check setup interfaces
    # Add box
    env.add_box(texture_path="",
                half_extent=to_nd_float_array([5.0, 5.0, 5.0]),
                pos=to_nd_float_array([10.0, 5.0, 10.0]),
                rot=0.0,
                detect_collision=False)

    # Add Sphere
    sphere_id = env.add_sphere(texture_path="",
                               radius=2.0,
                               pos=to_nd_float_array([0.0, 2.0, -5.0]),
                               rot=0.0,
                               detect_collision=True)
    print("sphere_id={}".format(sphere_id))

    # Locate agent
    env.locate_agent(pos=to_nd_float_array([0.0, 1.0, 0.0]),
                     rot=0.0)

    # Set light direction
    env.set_light_dir(dir=to_nd_float_array([0.0, -1.0, 0.0]))
    
    # Check step with action
    action = np.array([10, 0, 0], dtype=np.int32)

    for i in range(3):
      obs = env.step(action=action, num_steps=1)
    
    screen = obs["screen"]
    scipy.misc.imsave("debug.png", screen)

    # Check shape
    self.assertEqual( (width,height,3), screen.shape )

    # dtype should be uint8
    self.assertEqual(np.uint8, screen.dtype)

    collided = obs["collided"]

    print("collided size={}".format(len(collided)))

if __name__ == '__main__':
  unittest.main()
