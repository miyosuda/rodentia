# -*- coding: utf-8 -*-
import numpy as np
import pygame, sys
from pygame.locals import *

from aai_environment import AAIEnvironment

BLACK = (0, 0, 0)


class Display(object):
    def __init__(self, display_size):
        self.width = display_size[0]
        self.height = display_size[1]

        self.env = AAIEnvironment(256, 256)

        pygame.init()

        #self.surface = pygame.display.set_mode(display_size, 0, 24)
        self.surface = pygame.display.set_mode(display_size, 0, 32)
        pygame.display.set_caption('rodentia')

        self.last_state = self.env.reset()

    def update(self):
        self.surface.fill(BLACK)
        self.process()
        pygame.display.update()

    def get_real_action(self):
        """
        Unlike other examples, this example uses real action as array with 3 int values.
        [look, strafe, move] with WASD control.
        """
        lookAction = 0
        strafeAction = 0
        moveAction = 0

        pressed = pygame.key.get_pressed()

        if pressed[K_q]:
            lookAction += 6
        if pressed[K_e]:
            lookAction -= 6
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
        real_action = self.get_real_action()

        state, reward, terminal = self.env.step(real_action=real_action)

        top_image = self.env.get_top_view()

        if reward != 0:
            print("reward={}".format(reward))

        image = pygame.image.frombuffer(state, (256, 256), 'RGB')
        top_image = pygame.image.frombuffer(top_image, (256, 256), 'RGB')
        
        self.surface.blit(image, (0, 0))
        self.surface.blit(top_image, (256, 0))

        self.last_state = state

        if terminal:
            self.last_state = self.env.reset()

    def get_frame(self):
        data = self.surface.get_buffer().raw
        return data


def main():
    display_size = (512, 256)
    display = Display(display_size)
    clock = pygame.time.Clock()

    running = True
    FPS = 60

    recording = False
    
    if recording:
        from movie_writer import MovieWriter
        writer = MovieWriter("out.mov", display_size, FPS)
    else:
        writer = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        display.update()
        clock.tick(FPS)

        if writer is not None:
            frame_str = display.get_frame()
            d = np.fromstring(frame_str, dtype=np.uint8)
            d = d.reshape((display_size[1], display_size[0], 4))
            d = d[:,:,:3]
            writer.add_frame(d)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
