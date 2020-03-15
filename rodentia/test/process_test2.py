# -*- coding: utf-8 -*-
import unittest
import numpy as np
#from multiprocessing import Process, Pipe
from torch import multiprocessing as mp
import os, sys
sys.path.insert(0, os.getcwd())
import rodentia


COMMAND_ACTION    = 0
COMMAND_TERMINATE = 1

def worker(conn):
    env = rodentia.Environment(width=84, height=84,
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
            obs = env.step(action)
            conn.send(obs)
        elif command == COMMAND_TERMINATE:
            break
        else:
            print("bad command: {}".format(command))
  
    del env
    conn.send(0)
    conn.close()


class RodentiaProcessTest(unittest.TestCase):
  def testProcess(self):
      """
      env_dummy = rodentia.Environment(width=84, height=84,
                                       bg_color=[0.0, 0.0, 0.0])
      env_dummy.close()
      env_dummy = None
      """

      #use_context = False
      use_context = True

      conn0, child_conn0 = mp.Pipe()
      conn1, child_conn1 = mp.Pipe()

      if use_context:
          context = mp.get_context("fork")
          proc0 = context.Process(target=worker, args=(child_conn0,))
          proc1 = context.Process(target=worker, args=(child_conn1,))          
      else:
          proc0 = mp.Process(target=worker, args=(child_conn0,))
          proc1 = mp.Process(target=worker, args=(child_conn1,))
      
      proc0.start()
      proc1.start()
      
      conn0.recv()
      conn1.recv()
          
      conn0.send(COMMAND_ACTION)
      conn1.send(COMMAND_ACTION)
      
      obs0 = conn0.recv()
      obs1 = conn1.recv()

      screen0 = obs0["screen"]
      self.assertEqual( (84,84,3), screen0.shape )

      screen1 = obs1["screen"]
      self.assertEqual( (84,84,3), screen1.shape )
      
      conn0.send(COMMAND_TERMINATE)
      conn1.send(COMMAND_TERMINATE)



if __name__ == '__main__':
    unittest.main()
