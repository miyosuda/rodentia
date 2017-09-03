# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os
from collections import deque
import pygame, sys
from pygame.locals import *
import rodent_module

BLACK = (0, 0, 0)

MAX_STEP_NUM = 60 * 30

def to_nd_float_array(list_obj):
  return np.array(list_obj, dtype=np.float32)

class Display(object):
  def __init__(self, display_size):
    self.width  = 640
    self.height = 480
    self.env = rodent_module.Env(width=self.width, height=self.height)

    self.prepare_wall()

    self.obj_ids_set = set()
    
    self.reset()
    
    pygame.init()
    
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    pygame.display.set_caption('rodent')

  def prepare_wall(self):
    # -Z
    self.env.add_box(half_extent=to_nd_float_array([20.0, 1.0, 1.0]),
                     pos=to_nd_float_array([0.0, 1.0, -20.0]),
                     rot=0.0,
                     detect_collision=False)
    # +Z
    self.env.add_box(half_extent=to_nd_float_array([20.0, 1.0, 1.0]),
                     pos=to_nd_float_array([0.0, 1.0, 20.0]),
                     rot=0.0,
                     detect_collision=False)
    # -X
    self.env.add_box(half_extent=to_nd_float_array([1.0, 1.0, 20.0]),
                     pos=to_nd_float_array([-20.0, 1.0, 0.0]),
                     rot=0.0,
                     detect_collision=False)
    # +X
    self.env.add_box(half_extent=to_nd_float_array([1.0, 1.0, 20.0]),
                     pos=to_nd_float_array([20.0, 1.0, 0.0]),
                     rot=0.0,
                     detect_collision=False)

  def update(self):
    self.surface.fill(BLACK)
    self.process()
    pygame.display.update()

  def get_action(self):
    lookAction = 0
    strafeAction = 0
    moveAction = 0

    pressed = pygame.key.get_pressed()

    if pressed[K_q]:
      lookAction += 10
    if pressed[K_e]:
      lookAction -= 10
    if pressed[K_a]:
      strafeAction += 1
    if pressed[K_d]:
      strafeAction -= 1
    if pressed[K_w]:
      moveAction += 1
    if pressed[K_s]:
      moveAction -= 1
    return np.array([lookAction, strafeAction, moveAction],
                    dtype=np.int32)

  def process_sub(self, action):
    obs = self.env.step(action=action, num_steps=1)
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
    
  def process(self):
    action = self.get_action()
    screen, reward, terminal = self.process_sub(action)

    image = pygame.image.frombuffer(screen, (self.width,self.height), 'RGB')
    self.surface.blit(image, (0, 0))

    if terminal:
      self.reset()

  def clear_objects(self):
    for id in self.obj_ids_set:
      self.env.remove_obj(id)
    self.obj_ids_set = set()

  def reset(self):
    # Clear remaining reward objects
    self.clear_objects()
    
    # Reward Sphere
    obj_id0 = self.env.add_sphere(radius=1.0,
                                  pos=to_nd_float_array([-5.0, 1.0, -5.0]),
                                  rot=0.0,
                                  detect_collision=True)

    obj_id1 = self.env.add_sphere(radius=1.0,
                                  pos=to_nd_float_array([5.0, 1.0, -5.0]),
                                  rot=0.0,
                                  detect_collision=True)
    self.obj_ids_set.add(obj_id0)
    self.obj_ids_set.add(obj_id1)

    # add test model
    model_path0 = os.path.dirname(os.path.abspath(__file__)) + "/" + "../examples/data/apple0.obj"
    self.env.add_model(path=model_path0,
                       scale=to_nd_float_array([1.0, 1.0, 1.0]),
                       pos=to_nd_float_array([10.0, 0.0, 10.0]),
                       rot=0.0,
                       detect_collision=True)

    model_path1 = os.path.dirname(os.path.abspath(__file__)) + "/" + "../examples/data/lemon0.obj"
    self.env.add_model(path=model_path1,
                       scale=to_nd_float_array([1.0, 1.0, 1.0]),
                       pos=to_nd_float_array([-10.0, 0.0, 10.0]),
                       rot=0.0,
                       detect_collision=True)

    
    # Locate agent to default position
    self.env.locate_agent(pos=to_nd_float_array([0,0,0]),
                          rot=0.0)

    # Set light direction
    self.env.set_light_dir(dir=to_nd_float_array([-0.7,-1,-0.5]))

    self.total_reward = 0
    self.step_num = 0

def main():
  display_size = (640, 480)
  display = Display(display_size)
  clock = pygame.time.Clock()
  
  running = True
  FPS = 60

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          running = False
    
    display.update()
    clock.tick(FPS)
    
if __name__ == '__main__':
  main()
