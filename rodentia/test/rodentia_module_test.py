# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, os.getcwd())
import rodentia


def to_nd_float_array(list_obj):
    return np.array(list_obj, dtype=np.float32)


def imsave(path, image):
    pimage = Image.fromarray(image)
    pimage.save(path)
  

class RodentiaModuleTest(unittest.TestCase):
  def testVersion(self):
      version = rodentia.rodentia_module.version();
      self.assertEqual(version, rodentia.__version__)

  def testEnv(self):
      width  = 84 * 4
      height = 84 * 4

      env = rodentia.rodentia_module.Env()

      camera_id = env.add_camera_view(width, height,
                                      bg_color=to_nd_float_array([1.0, 0.0, 0.0]),
                                      near=0.05, far=80.0, focal_length=50.0,
                                      shadow_buffer_width=0)
      self.assertEqual(camera_id, 0)

      # Check setup interfaces
      # Add box
      env.add_box(texture_path="",
                  half_extent=to_nd_float_array([30.0, 1.0, 30.0]),
                  pos=to_nd_float_array([0.0, -1.0, 0.0]),
                  rot=to_nd_float_array([0.0, 0.0, 0.0, 1.0]),
                  mass=0.0,
                  detect_collision=False)

      # Add Sphere
      sphere_id = env.add_sphere(texture_path="",
                                 radius=1.0,
                                 pos=to_nd_float_array([0.0, 2.0, -5.0]),
                                 rot=to_nd_float_array([0.0, 0.0, 0.0, 1.0]),
                                 mass=1.0,
                                 detect_collision=True)
      print("sphere_id={}".format(sphere_id))

      # Locate agent
      env.locate_agent(pos=to_nd_float_array([0.0, 1.0, 0.0]),
                       rot_y=0.0)

      # Locate object
      env.locate_object(sphere_id,
                        pos=to_nd_float_array([0.0, 2.0, -6.0]),
                        rot=to_nd_float_array([0.0, 0.0, 0.0, 1.0]))
    
      # Set light parameters
      env.set_light(dir=to_nd_float_array([-1.0, -1.0, 0.0]),
                    color=to_nd_float_array([1.0, 1.0, 1.0]),
                    ambient_color=to_nd_float_array([0.4, 0.4, 0.4]),
                    shadow_rate=0.2)
    
      # Check step with action
      action = np.array([1, 0, 0], dtype=np.int32)
      
      for i in range(3):
          obs_step = env.step(action=action, num_steps=1)

      # Get object info
      info = env.get_agent_info()

      # Check shape
      self.assertEqual( (3,), info["pos"].shape )
      self.assertEqual( (3,), info["velocity"].shape )
      self.assertEqual( (4,), info["rot"].shape )

      obs_render = env.render(camera_id, info["pos"], info["rot"])

      screen = obs_render["screen"]
      imsave("debug.png", screen)

      # Check shape
      self.assertEqual( (width,height,3), screen.shape )

      # dtype should be uint8
      self.assertEqual(np.uint8, screen.dtype)

      collided = obs_step["collided"]
      print("collided size={}".format(len(collided)))

      # Get object info
      info = env.get_obj_info(sphere_id)

      # Check shape
      self.assertEqual( (3,), info["pos"].shape )
      self.assertEqual( (3,), info["velocity"].shape )
      self.assertEqual( (4,), info["rot"].shape )

      # Get agent info
      info = env.get_agent_info()

      # Check shape
      self.assertEqual( (3,), info["pos"].shape )
      self.assertEqual( (3,), info["velocity"].shape )
      self.assertEqual( (4,), info["rot"].shape )
      
      # Replace object texture
      texture_path = [""]
      env.replace_obj_texture(sphere_id, texture_path)
    

if __name__ == '__main__':
    unittest.main()
