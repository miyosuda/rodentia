# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rodent

MAX_STEP_NUM = 60 * 30

class SimpleEnvironment(object):
  ACTION_LIST = [
    [-20,   0,   0], # look_left
    [ 20,   0,   0], # look_right
    [  0,   -1,  0], # strafe_left
    [  0,    1,  0], # strafe_right
    [  0,    0,  1], # forward
    [  0,    0, -1], # backward
  ]

  def __init__(self, width, height):
    self.env = rodent.Environment(width=width, height=height)
    self._prepare_wall()
    
    self.obj_ids_set = set()
    self.reset()

  def get_action_size(self):
    return len(SimpleEnvironment.ACTION_LIST)

  def _prepare_wall(self):
    # -Z
    self.env.add_box(half_extent=[20.0, 1.0, 1.0],
                     pos=[0.0, 1.0, -20.0],
                     rot=0.0,
                     detect_collision=False)
    # +Z
    self.env.add_box(half_extent=[20.0, 1.0, 1.0],
                     pos=[0.0, 1.0, 20.0],
                     rot=0.0,
                     detect_collision=False)
    # -X
    self.env.add_box(half_extent=[1.0, 1.0, 20.0],
                     pos=[-20.0, 1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)
    # +X
    self.env.add_box(half_extent=[1.0, 1.0, 20.0],
                     pos=[20.0, 1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)
  
  def reset(self):
    # Clear remaining reward objects
    self._clear_objects()

    # Reward Sphere
    obj_id0 = self.env.add_sphere(radius=1.0,
                                  pos=[-5.0, 1.0, -5.0],
                                  rot=0.0,
                                  detect_collision=True)

    obj_id1 = self.env.add_sphere(radius=1.0,
                                  pos=[5.0, 1.0, -5.0],
                                  rot=0.0,
                                  detect_collision=True)

    self.obj_ids_set.add(obj_id0)
    self.obj_ids_set.add(obj_id1)
    
    # Locate agent to default position
    self.env.locate_agent(pos=[0,0,0],
                          rot=0.0)

    self.total_reward = 0
    self.step_num = 0
    obs = self.env.step(action=[0,0,0], num_steps=1)
    screen = obs["screen"]    
    return screen

  def _clear_objects(self):
    for id in self.obj_ids_set:
      self.env.remove_obj(id)
    self.obj_ids_set = set()

  def step(self, action):
    real_action = SimpleEnvironment.ACTION_LIST[action]

    obs = self.env.step(action=real_action, num_steps=1)
    self.step_num += 1
    
    screen = obs["screen"]
    collided = obs["collided"]

    reward = 0
    if len(collided) != 0:
      for id in collided:
        reward += 1
        self.env.remove_obj(id)

    self.total_reward += reward
    terminal = self.total_reward >= 2 or self.step_num >= MAX_STEP_NUM
    return screen, reward, terminal   
