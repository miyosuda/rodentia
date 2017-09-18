# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rodent
import os
import math
import random

MAX_STEP_NUM = 60 * 60 # 60 seconds * 60 frames


class BilliardEnvironment(object):
  def __init__(self, width, height):
    # Where model and texture data are located 
    self.data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/"

    # Create environment
    self.env = rodent.Environment(width=width, height=height,
                                  bg_color=[0,0,0])

    # Prepare stage objects
    self._prepare_stage()

    # Object id for collision checking
    self.ball_obj_id_list = []

    # Reset stage
    self.reset()
    
  def _prepare_stage(self):
    # Floor
    floor_texture_path = self.data_path + "floor1.png"

    self.env.add_box(texture_path=floor_texture_path,
                     half_extent=[17, 1.0, 17],
                     pos=[0.0, -1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)
    self.env.add_box(texture_path=floor_texture_path,
                     half_extent=[17, 1.0, 1.0],
                     pos=[0.0, -1.0, -18],
                     rot=0.0,
                     detect_collision=False)
    self.env.add_box(texture_path=floor_texture_path,
                     half_extent=[17, 1.0, 1.0],
                     pos=[0.0, -1.0, 18],
                     rot=0.0,
                     detect_collision=False)
    self.env.add_box(texture_path=floor_texture_path,
                     half_extent=[1.0, 1.0, 17],
                     pos=[-18, -1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)
    self.env.add_box(texture_path=floor_texture_path,
                     half_extent=[1.0, 1.0, 17],
                     pos=[18, -1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)

    # Wall
    wall_distance = 20.0
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

  def _locate_ball_obj(self, x, z):
    """ Locate ball object """
    texture_path = self.data_path + "checker.png"
    pos = [x, 1.0, z]
    # If the object's mass is not 0, the it is simulated as a rigid body object.
    ball_mass = 0.5
    obj_id = self.env.add_sphere(texture_path=texture_path,
                                 radius=1.0,
                                 pos=pos,
                                 rot=0.0,
                                 mass=ball_mass,
                                 detect_collision=False)
    self.ball_obj_id_list.append(obj_id)

  def _reset_sub(self):
    # First clear remaining reward objects
    self._clear_ball_objs()

    # Add reward objects
    self._locate_ball_obj(x=3, z=-3)
    self._locate_ball_obj(x=-4, z=2)
    
    # Locate agent to default position with randomized orientation
    rot = 2.0 * math.pi * random.random()
    
    self.env.locate_agent(pos=[0,1,0], rot=rot)

    # Reset environment and get screen
    obs = self.env.step(action=[0,0,0], num_steps=1)
    screen = obs["screen"]
    return screen
  
  def reset(self):
    self.step_num = 0
    return self._reset_sub()

  def _clear_ball_objs(self):
    # Create reward objects
    for id in self.ball_obj_id_list:
      self.env.remove_obj(id)
      
    self.ball_obj_id_list = []

  def _check_balls_in_pockets(self):
    """ Check wheter balls are fallen in the pocket. """
    reward = 0
    
    for id in self.ball_obj_id_list[:]:
      info = self.env.get_obj_info(id)
      pos = info["pos"]
      if pos[1] < -1.0:
        # If the ball is fallen in the packet.
        # Remove ball obj from env.
        self.env.remove_obj(id)
        # Remove ball's id from id list.
        self.ball_obj_id_list.remove(id)
        reward += 1
    return reward

  def _check_agent_in_pockets(self):
    """ Check whether agent is fallen in the packet or not. """
    
    info = self.env.get_agent_info()
    pos = info["pos"]
    return pos[1] < 0.0

  def step(self, real_action):
    obs = self.env.step(action=real_action, num_steps=1)
    self.step_num += 1
    
    screen = obs["screen"]

    # Check if balls are fallen in the pocket
    reward = self._check_balls_in_pockets()
    
    # Check if all positive rewards are taken
    is_empty = len(self.ball_obj_id_list) == 0

    agent_fallen = self._check_agent_in_pockets()

    # Episode ends when step size exceeds MAX_STEP_NUM
    terminal = self.step_num >= MAX_STEP_NUM or agent_fallen
    
    if (not terminal) and is_empty:
      screen = self._reset_sub()
    
    return screen, reward, terminal
