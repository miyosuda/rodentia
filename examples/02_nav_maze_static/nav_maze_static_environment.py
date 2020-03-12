# -*- coding: utf-8 -*-
import rodentia
import os
import math
import random

MAX_STEP_NUM = 60 * 60  # 60 seconds * 60 frames


class NavMazeStaticEnvironment(object):
    ACTION_LIST = [
        [  6,  0,  0], # look_left
        [ -6,  0,  0], # look_right
        [  0,  1,  0], # strafe_left
        [  0, -1,  0], # strafe_right
        [  0,  0,  1], # forward
        [  0,  0, -1], # backward
    ]    

    def __init__(self, width, height):
        self.data_path = os.path.dirname(
            os.path.abspath(__file__)) + "/../data/"

        self.env = rodentia.Environment(
            width=width, height=height, bg_color=[0.66, 0.91, 0.98])
        # Start pos
        self.start_pos_list = []
        self.start_pos_list.append([4, 0, 9])
        self.start_pos_list.append([4, 0, 7])
        self.start_pos_list.append([4, 0, -1])
        self.start_pos_list.append([4, 0, -3])
        self.start_pos_list.append([4, 0, -5])
        self.start_pos_list.append([4, 0, -7])
        self.start_pos_list.append([4, 0, -9])
        self.start_pos_list.append([2, 0, 9])
        self.start_pos_list.append([2, 0, 7])
        self.start_pos_list.append([2, 0, -1])
        self.start_pos_list.append([2, 0, -3])
        self.start_pos_list.append([2, 0, -5])
        self.start_pos_list.append([2, 0, -9])
        self.start_pos_list.append([0, 0, 9])
        self.start_pos_list.append([0, 0, 7])
        self.start_pos_list.append([0, 0, 1])
        self.start_pos_list.append([0, 0, -1])
        self.start_pos_list.append([0, 0, -3])
        self.start_pos_list.append([0, 0, -5])

        # Reward pos
        self.reward_pos_list = []
        self.reward_pos_list.append([2, 0.3, -7])
        self.reward_pos_list.append([0, 0.3, -7])
        self.reward_pos_list.append([0, 0.3, -9])
        self.reward_pos_list.append([-2, 0.3, 7])
        self.reward_pos_list.append([-2, 0.3, 9])

        # Reard obj ids
        self.reward_obj_ids_set = set()

        # Goal pos
        self.goal_pos = [4, 1, 1]

        # Prepare wall and floor and goal object.
        self._prepare_stage()

        # Reset scene
        self.reset()

    def get_action_size(self):
        return len(NavMazeStaticEnvironment.ACTION_LIST)

    def _prepare_stage(self):
        # Floor
        floor_texture_path = self.data_path + "floor1.png"

        self.env.add_box(
            texture_path=floor_texture_path,
            half_extent=[6.0, 1.0, 12.0],
            pos=[0.0, -1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        # Goal object (Red Sphere)
        goal_texture_path = self.data_path + "red.png"
        self.goal_obj_id = self.env.add_sphere(
            texture_path=goal_texture_path,
            radius=0.5,
            pos=self.goal_pos,
            rot=0.0,
            detect_collision=True)

        # Wall
        wall_texture_path = self.data_path + "wall1.png"
        wall_thickness = 0.1

        # [Center room]
        # -Z
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[0.0, 1.0, -4.0],
            rot=0.0,
            detect_collision=False)

        # +X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 3.0],
            pos=[1.0, 1.0, -1.0],
            rot=0.0,
            detect_collision=False)

        # +Z
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[2.0, 1.0, wall_thickness],
            pos=[-1.0, 1.0, 2.0],
            rot=0.0,
            detect_collision=False)

        # -X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 1.0],
            pos=[-3.0, 1.0, 1.0],
            rot=0.0,
            detect_collision=False)

        # -X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 1.0],
            pos=[-3.0, 1.0, -3.0],
            rot=0.0,
            detect_collision=False)

        # [Outer wall]
        # Left (-X) wall
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 10.0],
            pos=[-5.0, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        # Right (+X) wall
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 10.0],
            pos=[5.0, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        # -Z wall
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[5.0, 1.0, wall_thickness],
            pos=[0.0, 1.0, -10.0],
            rot=0.0,
            detect_collision=False)
        # +Z wall
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[5.0, 1.0, wall_thickness],
            pos=[0.0, 1.0, 10.0],
            rot=0.0,
            detect_collision=False)

        # [-Z L shape]
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[2.0, 1.0, wall_thickness],
            pos=[-1.0, 1.0, -6.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 2.0],
            pos=[-3.0, 1.0, -8.0],
            rot=0.0,
            detect_collision=False)

        # [-Z 7 shape]
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[2.0, 1.0, wall_thickness],
            pos=[1.0, 1.0, -8.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 3.0],
            pos=[3.0, 1.0, -5.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[2.0, 1.0, -2.0],
            rot=0.0,
            detect_collision=False)

        # Large panel beside goal
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[2.0, 1.0, wall_thickness],
            pos=[3.0, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)
        # Small panel beside goal
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[2.0, 1.0, 2.0],
            rot=0.0,
            detect_collision=False)

        # Chair shape
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[2.0, 1.0, 4.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 1.0],
            pos=[3.0, 1.0, 5.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[4.0, 1.0, 6.0],
            rot=0.0,
            detect_collision=False)

        # Bigge chair shape
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[2.0, 1.0, wall_thickness],
            pos=[-1.0, 1.0, 6.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 1.0],
            pos=[1.0, 1.0, 7.0],
            rot=0.0,
            detect_collision=False)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[2.0, 1.0, 8.0],
            rot=0.0,
            detect_collision=False)

        # Side line
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_thickness, 1.0, 2.0],
            pos=[-1.0, 1.0, 4.0],
            rot=0.0,
            detect_collision=False)

        # One panel below
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_thickness],
            pos=[-4.0, 1.0, 4.0],
            rot=0.0,
            detect_collision=False)

        # Twoe panels below
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[2.0, 1.0, wall_thickness],
            pos=[-3.0, 1.0, 8.0],
            rot=0.0,
            detect_collision=False)

    def _clear_reward_objects(self):
        for id in self.reward_obj_ids_set:
            self.env.remove_obj(id)
        self.reward_obj_ids_set = set()

    def _locate_reward_objects(self):
        # Clear remaining reward objects first
        self._clear_reward_objects()

        # Choose random 3 positions for rewards
        model_path = self.data_path + "apple0.obj"

        random.shuffle(self.reward_pos_list)

        for i in range(3):
            reward_pos = self.reward_pos_list[i]
            scale = 0.3
            obj_id = self.env.add_model(
                path=model_path,
                scale=[scale, scale, scale],
                pos=reward_pos,
                rot=0.0,
                detect_collision=True)
            # Store object id for collision checking
            self.reward_obj_ids_set.add(obj_id)

    def _locate_agent(self):
        # Choose random position and orientation for the agent.
        start_pos_index = random.randint(0, len(self.start_pos_list) - 1)
        start_pos = self.start_pos_list[start_pos_index]
        rot_y = 2.0 * math.pi * random.random()
        self.env.locate_agent(pos=start_pos, rot_y=rot_y)

    def _reset_sub(self):
        # Locate reward objects
        self._locate_reward_objects()

        # Locate agent
        self._locate_agent()

        # Reset environmenet and return screen image
        obs = self.env.step(action=[0, 0, 0])
        screen = obs["screen"]
        return screen

    def reset(self):
        self.step_num = 0
        return self._reset_sub()

    def step(self, action):
        #if action == -1:
        #  real_action = [0,0,0]
        #else:
        #  real_action = NavMazeStaticEnvironment.ACTION_LIST[action]
        real_action = NavMazeStaticEnvironment.ACTION_LIST[action]

        obs = self.env.step(action=real_action)
        self.step_num += 1

        screen = obs["screen"]
        collided = obs["collided"]  # ids for collided objects

        reward = 0
        goal_arrived = False

        # Check collision
        if len(collided) != 0:
            for id in collided:
                if id in self.reward_obj_ids_set:
                    # If collided with reward object
                    reward += 1
                    # Remove reward object
                    self.reward_obj_ids_set.remove(id)
                    self.env.remove_obj(id)
                elif id == self.goal_obj_id:
                    # If collided with goal object
                    reward += 10
                    goal_arrived = True

        # Episode ends when step size exceeds MAX_STEP_NUM
        terminal = self.step_num >= MAX_STEP_NUM

        if (not terminal) and goal_arrived:
            # Reset rewards and locate agent to random position
            screen = self._reset_sub()

        return screen, reward, terminal
