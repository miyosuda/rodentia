# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import os
from collections import deque
import pygame, sys
from pygame.locals import *
import rodent

BLACK = (0, 0, 0)

def to_nd_float_array(list_obj):
  return np.array(list_obj, dtype=np.float32)

class Display(object):
  def __init__(self, display_size):
    self.width  = 640
    self.height = 480
    self.env = rodent.Env(width=self.width, height=self.height)

    self.env.add_box(half_extent=to_nd_float_array([5.0, 10.0, 5.0]),
                     pos=to_nd_float_array([10.0, 5.0, 10.0]),
                     rot=0.0,
                     detect_collision=False)

    sphere_id = self.env.add_sphere(radius=2.0,
                                    pos=to_nd_float_array([-5.0, 2.0, -5.0]),
                                    rot=0.0,
                                    detect_collision=True)
    
    pygame.init()
    
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    pygame.display.set_caption('rodent')

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

  def process(self):
    action = self.get_action()
    obs = self.env.step(action=action)
    screen = obs["screen"]
    image = pygame.image.frombuffer(screen, (self.width,self.height), 'RGBA')
    self.surface.blit(image, (0, 0))

    collided = obs["collided"]
    if len(collided) != 0:
      for id in collided:
        self.env.remove_obj(id)

def main(args):
  display_size = (640, 480)
  display = Display(display_size)
  clock = pygame.time.Clock()
  
  running = True
  FPS = 60

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
    
    display.update()
    clock.tick(FPS)
    
    
if __name__ == '__main__':
  tf.app.run()
