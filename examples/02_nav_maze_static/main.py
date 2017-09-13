# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pygame, sys
from pygame.locals import *

from nav_maze_static_environment import NavMazeStaticEnvironment
from movie_writer import MovieWriter

BLACK = (0, 0, 0)


class RandomAgent(object):
  def __init__(self, action_num):
    self.action_num = action_num

  def choose_action(self, state):
    return np.random.randint(self.action_num)


class Display(object):
  def __init__(self, display_size):
    self.width = display_size[0]
    self.height = display_size[1]
    
    self.env = NavMazeStaticEnvironment(width=self.width, height=self.height)
    self.agent = RandomAgent(self.env.get_action_size())

    pygame.init()
    
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    pygame.display.set_caption('rodent')

    self.last_state = self.env.reset()

  def update(self):
    self.surface.fill(BLACK)
    self.process()
    pygame.display.update()

  def get_real_action(self):
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
    return [lookAction, strafeAction, moveAction]

  def process(self):
    #action = self.agent.choose_action(self.last_state)
    #state, reward, terminal = self.env.step(action=action)

    real_action = self.get_real_action()
    state, reward, terminal = self.env.step(real_action=real_action)

    if reward != 0:
      print("reward={}".format(reward))
    
    image = pygame.image.frombuffer(state, (self.width, self.height), 'RGB')
    self.surface.blit(image, (0, 0))

    self.last_state = state

    if terminal:
      self.last_state = self.env.reset()

  def get_frame(self):
    data = self.surface.get_buffer().raw
    return data

      
def main():
  display_size = (640, 480)
  display = Display(display_size)
  clock = pygame.time.Clock()
  
  running = True
  FPS = 60

  writer = MovieWriter("navmaze.mov", display_size, FPS)

  while running:
    frame_str = display.get_frame()
    d = np.fromstring(frame_str, dtype=np.uint8)
    d = d.reshape((display_size[1], display_size[0], 3))
    writer.add_frame(d)
    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          running = False
    
    display.update()
    clock.tick(FPS)

  writer.close()
    
if __name__ == '__main__':
  main()
