# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import scipy.misc
import sys
sys.path.insert(0, os.getcwd())
import rodent


def to_nd_float_array(list_obj):
  return np.array(list_obj, dtype=np.float32)
  

class RodentModuleTest(unittest.TestCase):
  def testVersion(self):
    version = rodent.rodent_module.version();
    self.assertEqual(version, "0.1.0")

  def testEnv(self):
    width  = 84 * 4
    height = 84 * 4
    env = rodent.rodent_module.Env(width=width, height=height,
                                   bg_color=to_nd_float_array([1.0, 1.0, 1.0]))

    # Check setup interfaces
    # Add box
    env.add_box(texture_path="",
                half_extent=to_nd_float_array([30.0, 1.0, 30.0]),
                pos=to_nd_float_array([0.0, -1.0, 0.0]),
                rot=0.0,
                mass=0.0,
                detect_collision=False)

    # Add Sphere
    sphere_id = env.add_sphere(texture_path="",
                               radius=1.0,
                               pos=to_nd_float_array([0.0, 2.0, -5.0]),
                               rot=0.0,
                               mass=1.0,
                               detect_collision=True)
    print("sphere_id={}".format(sphere_id))

    # Locate agent
    env.locate_agent(pos=to_nd_float_array([0.0, 1.0, 0.0]),
                     rot=0.0)

    # Locate object
    env.locate_object(sphere_id,
                      pos=to_nd_float_array([0.0, 1.0, -1.0]),
                      rot=0.0)

    # Set light direction
    env.set_light_dir(dir=to_nd_float_array([-1.0, -1.0, 0.0]))
    
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

    # Get object info
    info = env.get_obj_info(sphere_id)

    # Check shape
    self.assertEqual( (3,), info["pos"].shape )
    self.assertEqual( (3,), info["velocity"].shape )
    self.assertEqual( (3,), info["euler_angles"].shape )

    # Get agent info
    info = env.get_agent_info()

    # Check shape
    self.assertEqual( (3,), info["pos"].shape )
    self.assertEqual( (3,), info["velocity"].shape )
    self.assertEqual( (3,), info["euler_angles"].shape )
    

if __name__ == '__main__':
  unittest.main()
