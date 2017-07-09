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

class Display(object):
  def __init__(self, display_size):
    self.width  = 640
    self.height = 480
    self.env = rodent.Env(width=self.width, height=self.height)
    pygame.init()
    
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    pygame.display.set_caption('rodent')

  def update(self):
    self.surface.fill(BLACK)
    self.process()
    pygame.display.update()

  def process(self):
    action = np.array([0, 0, 0], dtype=np.int32)
    obs = self.env.step(action=action)
    screen = obs["screen"]
    image = pygame.image.frombuffer(screen, (self.width,self.height), 'RGBA')
    self.surface.blit(image, (0, 0))    

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
