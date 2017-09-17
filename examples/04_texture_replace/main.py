# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import scipy.misc
import math
##..
#import sys
#sys.path.insert(0, os.getcwd())
##..
import rodent



IMAGE_SIZE = 84


def main():
  data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/"

  env = rodent.Environment(width=IMAGE_SIZE, height=IMAGE_SIZE,
                           bg_color=[0.66, 0.91, 0.98])



  floor_texture_path = data_path + "floor0.png"

  floor_id = env.add_box(texture_path=floor_texture_path,
                         half_extent=[30.0, 1.0, 30.0],
                         pos=[0.0, -1.0, 0.0])

  wall_texture_path = data_path + "wall0.png"

  wall_id0 = env.add_box(texture_path=wall_texture_path,
                         half_extent=[3.0, 1.0, 1.0],
                         pos=[0.0, 1.0, -20.0])

  wall_id1 = env.add_box(texture_path=wall_texture_path,
                         half_extent=[0.01, 1.0, 10.0],
                         pos=[1.0, 1.0, -10.0])

  wall_id2 = env.add_box(texture_path=wall_texture_path,
                         half_extent=[0.01, 1.0, 10.0],
                         pos=[-1.0, 1.0, -10.0])

  #model_path0 = data_path + "ice_lolly0.obj"
  model_path0 = data_path + "hat0.obj"
  #model_path0 = data_path + "suitcase0.obj"
  rot = math.pi * 0.5
  
  obj_id0 = env.add_model(path=model_path0,
                          scale=[0.8, 0.8, 0.8],
                          pos=[0,1.5,-5],
                          rot=rot)

  env.set_light(dir=[0.0, -1.0, -1.0],
                color=[0.4, 0.4, 0.4],
                ambient_color=[0.9, 0.9, 0.9],
                shadow_rate=0.8)

  action = [0, 0, 0]
  obs = env.step(action=action)
  obs = env.step(action=action)

  screen = obs["screen"]

  scipy.misc.imsave("debug.png", screen)

  

if __name__ == '__main__':
  main()
