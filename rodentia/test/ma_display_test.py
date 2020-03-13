import numpy as np
import os
import pygame, sys
from pygame.locals import *
import sys
sys.path.insert(0, os.getcwd())
import rodentia

BLACK = (0, 0, 0)

MAX_STEP_NUM = 60 * 30


class Display(object):
    def __init__(self, display_size):
        self.width = 256
        self.height = 256

        self.data_path = os.path.dirname(
            os.path.abspath(__file__)) + "/../../examples/data/"

        self.env = rodentia.MultiAgentEnvironment(agent_size=2,
                                                  width=self.width,
                                                  height=self.height,
                                                  bg_color=[0.1, 0.1, 0.1])

        self.prepare_stage()

        self.obj_ids_set = set()

        self.reset()

        pygame.init()

        self.surface = pygame.display.set_mode(display_size, 0, 24)
        pygame.display.set_caption('rodentia')

    def prepare_stage(self):
        floor_texture_path = self.data_path + "floor3.png"

        # Floor
        self.env.add_box(
            texture_path=floor_texture_path,
            half_extent=[20.0, 1.0, 20.0],
            pos=[0.0, -1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        wall_texture_path = self.data_path + "wall2.png"

        # -Z
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[20.0, 1.0, 1.0],
            pos=[0.0, 1.0, -20.0],
            rot=0.0,
            detect_collision=False)
        # +Z
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[20.0, 1.0, 1.0],
            pos=[0.0, 1.0, 20.0],
            rot=0.0,
            detect_collision=False)
        # -X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, 20.0],
            pos=[-20.0, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)
        # +X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, 20.0],
            pos=[20.0, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        # Debug box
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, 1.0],
            pos=[0.0, 1.0, -5.0],
            rot=0,
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
        return np.array([lookAction, strafeAction, moveAction], dtype=np.int32)

    def process_sub(self, action):
        obs = self.env.step(action=action)
        self.step_num += 1

        screen = obs["screen"]
        collided = obs["collided"]

        terminal = self.step_num >= MAX_STEP_NUM
        reward = 0
        return screen, reward, terminal

    def process(self):
        action = self.get_action()
        actions = [action, [0,0,0]]
        screen, reward, terminal = self.process_sub(actions)

        image0 = pygame.image.frombuffer(screen[0], (self.width, self.height),
                                        'RGB')
        image1 = pygame.image.frombuffer(screen[1], (self.width, self.height),
                                        'RGB')
        self.surface.blit(image0, (0, 0))
        self.surface.blit(image1, (self.width, 0))

        if terminal:
            self.reset()

    def clear_objects(self):
        for id in self.obj_ids_set:
            self.env.remove_obj(id)
        self.obj_ids_set = set()

    def reset(self):
        # Clear remaining reward objects
        self.clear_objects()

        texture_path = self.data_path + "red.png"

        # Reward Sphere
        obj_id0 = self.env.add_sphere(
            texture_path=texture_path,
            radius=1.0,
            pos=[-5.0, 1.0, 5.0],
            rot=0.0,
            mass=1.0,
            detect_collision=True)

        obj_id1 = self.env.add_sphere(
            texture_path=texture_path,
            radius=1.0,
            pos=[5.0, 1.0, 5.0],
            rot=0.0,
            mass=1.0,
            detect_collision=True)
        self.obj_ids_set.add(obj_id0)
        self.obj_ids_set.add(obj_id1)

        # add test model
        model_path0 = self.data_path + "apple0.obj"
        self.env.add_model(
            path=model_path0,
            scale=[1.0, 1.0, 1.0],
            pos=[0.0, 0.0, 10.0],  # +z pos
            rot=0.0,
            mass=1.0,
            detect_collision=True)

        model_path1 = self.data_path + "lemon0.obj"
        self.env.add_model(
            path=model_path1,
            scale=[1.0, 1.0, 1.0],
            pos=[10.0, 0.0, 10.0],
            rot=0.0,
            mass=1.0,
            detect_collision=True)

        model_path2 = self.data_path + "ramp0.obj"
        self.env.add_model(
            path=model_path2,
            scale=[2.0, 1.0, 2.0],
            pos=[10.0, 0.0, 5.0],
            rot=np.pi * 0.25,
            mass=0.0,
            detect_collision=False,
            use_mesh_collision=True)

        model_path3 = self.data_path + "cylinder0.obj"
        self.env.add_model(
            path=model_path3,
            scale=[3.0, 3.0, 3.0],
            pos=[-5.0, 0.0, 8.0],
            rot=0.0,
            mass=0.0,
            detect_collision=False,
            use_mesh_collision=True)

        # Locate agent to default position
        self.env.locate_agent(agent_index=0, pos=[0, 0, 0], rot_y=0.0)
        self.env.locate_agent(agent_index=0, pos=[0, 0, 2], rot_y=0.0)

        # Set light params
        self.env.set_light(dir=[-0.5, -1.0, -0.4])

        self.step_num = 0


def main():
    display_size = (512, 256)
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
