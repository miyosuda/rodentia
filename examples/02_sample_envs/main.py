import gym
import numpy as np
import pygame, sys
from pygame.locals import *


from gym.envs.registration import register
register(id='SeekAvoidArena-v0',
         entry_point='envs.seekavoid_environment:SeekAvoidEnvironment')
register(id='NavMazeStatic-v0',
         entry_point='envs.nav_maze_static_environment:NavMazeStaticEnvironment')
register(id='Billiard-v0',
         entry_point='envs.billiard_environment:BilliardEnvironment')
register(id='AnimalAI-v0',
         entry_point='envs.aai_environment:AAIEnvironment')



BLACK = (0, 0, 0)


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state):
        return self.action_space.sample()


class Display(object):
    def __init__(self, env_name, display_size, env_size, manual):
        self.width = display_size[0]
        self.height = display_size[1]

        self.manual = manual

        env_width = env_size[0]
        env_height = env_size[1]
        
        self.env = gym.make(env_name, width=env_width, height=env_height)
        
        self.agent = RandomAgent(self.env.action_space)

        pygame.init()

        self.surface = pygame.display.set_mode(display_size, 0, 24)
        pygame.display.set_caption('rodentia')

        self.last_state = self.env.reset()

    def update(self):
        self.surface.fill(BLACK)
        self.process()
        pygame.display.update()

    def get_manual_action(self):
        """
        Unlike other examples, this example uses real action as array with 3 int values.
        [look, strafe, move] with WASD control.
        """
        look_action = 1
        strafe_action = 1
        move_action = 1

        pressed = pygame.key.get_pressed()

        if pressed[K_q]:
            look_action += 1
        if pressed[K_e]:
            look_action -= 1
        if pressed[K_a]:
            strafe_action += 1
        if pressed[K_d]:
            strafe_action -= 1
        if pressed[K_w]:
            move_action += 1
        if pressed[K_s]:
            move_action -= 1
        return [look_action, strafe_action, move_action]

    def process(self):
        if self.manual:
            action = self.get_manual_action()
        else:
            action = self.agent.choose_action(self.last_state)
        
        state, reward, terminal, _ = self.env.step(action=action)

        #top_image = self.env.get_top_view()

        if reward != 0:
            print("reward={}".format(reward))

        state_h = state.shape[0]            
        state_w = state.shape[1]
        
        image = pygame.image.frombuffer(state, (state_w, state_h), 'RGB')

        #top_image = pygame.image.frombuffer(top_image, (256, 256), 'RGB')
        
        self.surface.blit(image, (0, 0))

        self.last_state = state

        if terminal:
            self.last_state = self.env.reset()

    def get_frame(self):
        data = self.surface.get_buffer().raw
        return data


def main():
    #env_name = "SeekAvoidArena-v0"
    #env_name = "NavMazeStatic-v0"
    #env_name = "Billiard-v0"
    env_name = "AnimalAI-v0"
    
    display_size = (256, 256)
    env_size = (256, 256)

    manual = True

    display = Display(env_name, display_size, env_size, manual)
    clock = pygame.time.Clock()

    running = True
    FPS = 60

    recording = True
    
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
