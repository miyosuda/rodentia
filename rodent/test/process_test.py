# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from multiprocessing import Process, Pipe
import os, sys
sys.path.insert(0, os.getcwd())
import rodent


COMMAND_ACTION    = 0
COMMAND_TERMINATE = 1

def worker(conn):
  env = rodent.Environment(width=84, height=84,
                           bg_color=[0.0, 0.0, 0.0])

  # Add floor
  env.add_box(texture_path="",
              half_extent=[10.0, 1.0, 10.0],
              pos=[0.0, -1.0, 0.0],
              rot=0.0,
              detect_collision=False)

  # Add sphere
  sphere_id = env.add_sphere(texture_path="",
                             radius=1.0,
                             pos=[0.0, 2.0, -5.0],
                             rot=0.0,
                             detect_collision=True)
  conn.send(0)
  
  while True:
    command = conn.recv()

    if command == COMMAND_ACTION:
      action = np.array([10, 0, 0], dtype=np.int32)
      obs = env.step(action, num_steps=1)
      conn.send(obs)
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
  
  del env
  conn.send(0)
  conn.close()

class RodentProcessTest(unittest.TestCase):
  def testProcess(self):
    conn, child_conn = Pipe()
    proc = Process(target=worker, args=(child_conn,))
    proc.start()
    conn.recv()

    conn.send(COMMAND_ACTION)
    obs = conn.recv()

    screen = obs["screen"]
    self.assertEqual( (84,84,3), screen.shape )

    conn.send(COMMAND_TERMINATE)

if __name__ == '__main__':
  unittest.main()
