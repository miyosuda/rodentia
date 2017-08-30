# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pygame, sys
from pygame.locals import *

from simple_environment import SimpleEnvironment

BLACK = (0, 0, 0)

class RandomAgent(object):
  def __init__(self, action_num):
    self.action_num = action_num

  def choose_action(self):
    return np.random.randint(self.action_num)

class Display(object):
  def __init__(self, display_size):
    self.width = 84
    self.height = 84
    
    self.env = SimpleEnvironment(width=self.width, height=self.height)
    self.agent = RandomAgent(self.env.get_action_size())

    self.env.reset()
    
    pygame.init()
    
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    pygame.display.set_caption('rodent')

  def update(self):
    self.surface.fill(BLACK)
    self.process()
    pygame.display.update()

  def process(self):
    action = self.agent.choose_action()
    
    state, reward, terminal = self.env.step(action=action)

    if reward != 0:
      print("reward={}".format(reward))
    
    #self.total_reward += reward

    image = pygame.image.frombuffer(state, (self.width, self.height), 'RGB')
    self.surface.blit(image, (0, 0))

    if terminal:
      #self.total_reward = 0
      self.env.reset()

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
    
    display.update()
    clock.tick(FPS)
    
if __name__ == '__main__':
  main()
