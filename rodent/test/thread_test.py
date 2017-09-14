# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import threading
import scipy.misc
import os, sys
sys.path.insert(0, os.getcwd())
import rodent

PARALLEL_SIZE = 8

def thread_function(parallel_index):
  bg_color = [0.0, 0.0, 0.0]
  if parallel_index % 2 == 0:
    bg_color = [1.0, 0.0, 0.0]

  env = rodent.Environment(width=84, height=84,
                           bg_color=bg_color)

  # Add floor
  env.add_box(texture_path="",
              half_extent=[10.0, 1.0, 10.0],
              pos=[0.0, -1.0, 0.0],
              rot=0.0,
              detect_collision=False)

  # Add sphere
  env.add_sphere(texture_path="",
                 radius=1.0,
                 pos=[0.0, 1.0, -5.0],
                 rot=0.0,
                 detect_collision=True)

  for i in range(1000):
    action = np.array([10, 0, 0], dtype=np.int32)
    obs = env.step(action, num_steps=1)

  screen = obs["screen"]
  file_name = "thread_image{}.png".format(parallel_index)
  scipy.misc.imsave(file_name, screen)
    
  del env

class RodentThreadTest(unittest.TestCase):
  def testProcess(self):
    threads = []
    for i in range(PARALLEL_SIZE):
      threads.append(threading.Thread(target=thread_function, args=(i,)))
      
    for t in threads:
      t.start()

    for t in threads:
      t.join()
      
    print("thread test finished")

if __name__ == '__main__':
  unittest.main()
