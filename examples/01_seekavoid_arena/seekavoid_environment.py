# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rodent
import os
import math
import random


MAX_STEP_NUM = 60 * 20 # 20 seconds * 60 frames 


class SeekAvoidEnvironment(object):
  ACTION_LIST = [
    [ 10,  0,  0], # look_left
    [-10,  0,  0], # look_right
    [  0,  1,  0], # strafe_left
    [  0, -1,  0], # strafe_right
    [  0,  0,  1], # forward
    [  0,  0, -1], # backward
  ]

  def __init__(self, width, height):
    # Where model and texture data are located 
    self.data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/"

    # Create environment
    self.env = rodent.Environment(width=width, height=height,
                                  bg_color=[0.66, 0.91, 0.98])

    # Prepare stage objects
    self._prepare_stage()

    # Object id for collision checking
    self.plus_obj_ids_set = set()
    self.minus_obj_ids_set = set()

    # Reset stage
    self.reset()
    
  def get_action_size(self):
    return len(SeekAvoidEnvironment.ACTION_LIST)

  def _prepare_stage(self):
    # Floor
    floor_texture_path = self.data_path + "floor1.png"

    self.env.add_box(texture_path=floor_texture_path,
                     half_extent=[30.0, 1.0, 30.0],
                     pos=[0.0, -1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)

    # Wall
    wall_distance = 30.0

    wall_texture_path = self.data_path +  "wall1.png"
    
    # -Z
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[wall_distance, 1.0, 1.0],
                     pos=[0.0, 1.0, -wall_distance],
                     rot=0.0,
                     detect_collision=False)
    # +Z
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[wall_distance, 1.0, 1.0],
                     pos=[0.0, 1.0, wall_distance],
                     rot=0.0,
                     detect_collision=False)
    # -X
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[1.0, 1.0, wall_distance],
                     pos=[-wall_distance, 1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)
    # +X
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[1.0, 1.0, wall_distance],
                     pos=[wall_distance, 1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)

  def _locate_plus_reward_obj(self, x, z, rot):
    """ Locate positive reward object """
    model_path = self.data_path + "apple0.obj"
    pos_scale = 0.075
    pos = [x * pos_scale, 0.0, z * pos_scale]
    obj_id = self.env.add_model(path=model_path,
                                scale=[1.0, 1.0, 1.0],
                                pos=pos,
                                rot=rot,
                                detect_collision=True)
    self.plus_obj_ids_set.add(obj_id)

  def _locate_minus_reward_obj(self, x, z, rot):
    """ Locate negative reward object """
    model_path = self.data_path + "lemon0.obj"
    pos_scale = 0.075
    pos = [x * pos_scale, 0.0, z * pos_scale]
    obj_id = self.env.add_model(path=model_path,
                                scale=[1.0, 1.0, 1.0],
                                pos=pos,
                                rot=rot,
                                detect_collision=True)
    self.minus_obj_ids_set.add(obj_id)

  def _reset_sub(self):
    # First clear remaining reward objects
    self._clear_objects()

    # Add reward objects
    self._locate_plus_reward_obj(x=96, z=0, rot=0.625)
    self._locate_plus_reward_obj(x=192, z=-112, rot=0.375)
    self._locate_plus_reward_obj(x=-128, z=-32, rot=0.0)
    self._locate_plus_reward_obj(x=-144, z=184, rot=0.0)
    self._locate_plus_reward_obj(x=176, z=208, rot=0.75)
    self._locate_plus_reward_obj(x=160, z=104, rot=0.0)
    self._locate_plus_reward_obj(x=80, z=192, rot=0.5)
    self._locate_plus_reward_obj(x=-120, z=-160, rot=0.375)
    self._locate_plus_reward_obj(x=-248, z=80, rot=0.5)
    self._locate_plus_reward_obj(x=96, z=-184, rot=0.0)
    self._locate_plus_reward_obj(x=-64, z=272, rot=0.125)
    self._locate_plus_reward_obj(x=288, z=-88, rot=0.875)
    self._locate_plus_reward_obj(x=-312, z=-96, rot=0.125)
    self._locate_plus_reward_obj(x=-256, z=-312, rot=0.875)
    self._locate_plus_reward_obj(x=240, z=232, rot=0.0)

    self._locate_minus_reward_obj(x=-184, z=-232, rot=0.0)
    self._locate_minus_reward_obj(x=104, z=-80, rot=0.0)
    self._locate_minus_reward_obj(x=200, z=40, rot=0.25)
    self._locate_minus_reward_obj(x=-240, z=-8, rot=0.625)
    self._locate_minus_reward_obj(x=-48, z=152, rot=0.0)
    self._locate_minus_reward_obj(x=48, z=-296, rot=0.875)
    self._locate_minus_reward_obj(x=-248, z=216, rot=0.375)
    
    # Locate agent to default position with randomized orientation
    rot = 2.0 * math.pi * random.random()
    
    self.env.locate_agent(pos=[0,0,0],
                          rot=rot)

    # Reset environment and get screen
    obs = self.env.step(action=[0,0,0], num_steps=1)
    screen = obs["screen"]
    return screen
  
  def reset(self):
    self.step_num = 0
    return self._reset_sub()

  def _clear_objects(self):
    # Create reward objects
    for id in self.plus_obj_ids_set:
      self.env.remove_obj(id)
    for id in self.minus_obj_ids_set:
      self.env.remove_obj(id)
      
    self.plus_obj_ids_set = set()
    self.minus_obj_ids_set = set()

  def step(self, action):
    # Get action value to set to environment
    real_action = SeekAvoidEnvironment.ACTION_LIST[action]

    obs = self.env.step(action=real_action, num_steps=1)
    self.step_num += 1
    
    screen = obs["screen"]
    collided = obs["collided"]

    # Check collision
    reward = 0
    if len(collided) != 0:
      for id in collided:
        if id in self.plus_obj_ids_set:
          reward += 1
          self.plus_obj_ids_set.remove(id)
        elif id in self.minus_obj_ids_set:
          reward -= 1
          self.minus_obj_ids_set.remove(id)
        # Remove reward object from environment
        self.env.remove_obj(id)

    # Check if all positive rewards are taken
    is_empty = len(self.plus_obj_ids_set) == 0

    # Episode ends when step size exceeds MAX_STEP_NUM
    terminal = self.step_num >= MAX_STEP_NUM
    
    if (not terminal) and is_empty:
      screen = self._reset_sub()
    
    return screen, reward, terminal
