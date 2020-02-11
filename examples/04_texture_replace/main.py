# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from PIL import Image
import math
import random
import rodent


# Width and height of the image
IMAGE_SIZE = 80

# Color patterns
colors = [
  [249.0/255.0, 248.0/255.0, 248.0/255.0],
  [64.0/255.0, 80.0/255.0, 142.0/255.0],
  [85.0/255.0, 86.0/255.0, 146.0/255.0],
  [143.0/255.0, 206.0/255.0, 208.0/255.0],
  [164.0/255.0, 207.0/255.0, 153.0/255.0],
  [118.0/255.0, 84.0/255.0, 144.0/255.0],
  [157.0/255.0, 208.0/255.0, 184.0/255.0],
  [158.0/255.0, 158.0/255.0, 156.0/255.0],
  [174.0/255.0, 129.0/255.0, 109.0/255.0],
  [168.0/255.0, 78.0/255.0, 140.0/255.0],
  [217.0/255.0, 60.0/255.0, 64.0/255.0],
  [216.0/255.0, 59.0/255.0, 134.0/255.0],
  [237.0/255.0, 189.0/255.0, 108.0/255.0],
  [241.0/255.0, 200.0/255.0, 204.0/255.0],
  [229.0/255.0, 231.0/255.0, 141.0/255.0],
  [237.0/255.0, 233.0/255.0, 133.0/255.0]
]

def imsave(path, image):
  pimage = Image.fromarray(image)
  pimage.save(path)
  
  
def imread(path):
  return np.array(Image.open(path))


def create_colored_image(color, image):
  """ Convert grayscale image to colored image. """
  color = color + [1.0]
  colored_image = image.astype(np.float32) * color
  return colored_image.astype(np.uint8)


def create_colored_images(data_path,
                          output_data_path,
                          base_name):
  """ Create and save colored images. """
  if not os.path.exists(output_data_path):
    os.mkdir(output_data_path)
  
  base_image = imread(data_path + base_name + "0.png")
  for i in range(len(colors)):
    color = colors[i]
    colored_image = create_colored_image(color, base_image)
    output_file_path = "{}/{}{}.png".format(output_data_path, base_name, i)
    imsave(output_file_path, colored_image)

    
def generate_model_images(model_base_name,
                          env,
                          data_path,
                          color_data_path,
                          generate_data_path,
                          floor_id,
                          wall_ids,
                          index_offset):
  """ Create dataset images with one .obj model data. """  

  if not os.path.exists(generate_data_path):
    os.mkdir(generate_data_path)
  
  model_path = data_path + model_base_name + "0.obj"

  obj_id = env.add_model(path=model_path,
                         scale=[0.8, 0.8, 0.8],
                         pos=[0, 1.5, -5],
                         rot=0.0)

  obs = env.step(action=[0,0,0])

  color_size = len(colors)

  for floor_color_index in range(color_size):
    # Change floor color
    floor_texture_path =  "{}/floor{}.png".format(color_data_path, floor_color_index)
    env.replace_obj_texture(floor_id, floor_texture_path)

    for wall_color_index in range(color_size):
      # Change wall color
      wall_texture_path =  "{}/wall{}.png".format(color_data_path, wall_color_index)

      for wall_id in wall_ids:
        env.replace_obj_texture(wall_id, wall_texture_path)
      
      for obj_color_index in range(color_size):
        # Change obj color
        obj_texture_path =  "{}/{}{}.png".format(color_data_path,
                                                 model_base_name,
                                                 obj_color_index)

        env.replace_obj_texture(obj_id, obj_texture_path)
        
        index = floor_color_index * (16*16) + \
                wall_color_index * 16 + \
                obj_color_index + \
                index_offset

        # Randomize object rotation and position
        rot = 2.0 * math.pi * random.random()
        distance = -5.9 + random.random()
        
        env.locate_object(obj_id,
                          pos=[0, 1.2, distance],
                          rot=rot)
        
        obs = env.step(action=[0,0,0])
        
        screen = obs["screen"]
        file_path = "{}/image{}.png".format(generate_data_path, index)
        imsave(file_path, screen)

  env.remove_obj(obj_id)


def main():
  # Where original texture and model data are located
  data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/"

  # Where converted textures are located
  output_data_path = os.path.dirname(os.path.abspath(__file__)) + "/colored_data"

  # Prepare color textures
  print("Start generating color variations of model textures")
  
  create_colored_images(data_path, output_data_path, "ice_lolly")
  create_colored_images(data_path, output_data_path, "hat")
  create_colored_images(data_path, output_data_path, "suitcase")
  create_colored_images(data_path, output_data_path, "wall")
  create_colored_images(data_path, output_data_path, "floor")

  # Create environment
  env = rodent.Environment(width=IMAGE_SIZE, height=IMAGE_SIZE,
                           bg_color=[0.66, 0.91, 0.98])

  # Prepare wall and floor objects
  floor_texture_path = data_path + "floor1.png"
  floor_id = env.add_box(texture_path=floor_texture_path,
                         half_extent=[30.0, 1.0, 30.0],
                         pos=[0.0, -1.0, 0.0])

  wall_texture_path = data_path + "wall1.png"
  wall_id0 = env.add_box(texture_path=wall_texture_path,
                         half_extent=[3.0, 1.0, 1.0],
                         pos=[0.0, 1.0, -20.0])

  wall_id1 = env.add_box(texture_path=wall_texture_path,
                         half_extent=[0.01, 1.0, 10.0],
                         pos=[1.0, 1.0, -10.0])

  wall_id2 = env.add_box(texture_path=wall_texture_path,
                         half_extent=[0.01, 1.0, 10.0],
                         pos=[-1.0, 1.0, -10.0])

  env.set_light(dir=[0.0, -1.0, -1.0],
                color=[0.4, 0.4, 0.4],
                ambient_color=[0.9, 0.9, 0.9],
                shadow_rate=0.8)

  # Where generated dataset images will be located
  generate_data_path = os.path.dirname(os.path.abspath(__file__)) + "/generated_data"

  wall_ids = [wall_id0, wall_id1, wall_id2]

  print("Start generating image datasets")

  generate_model_images("ice_lolly",
                        env,
                        data_path,
                        output_data_path,
                        generate_data_path,
                        floor_id,
                        wall_ids,
                        0)
  
  generate_model_images("hat",
                        env,
                        data_path,
                        output_data_path,
                        generate_data_path,
                        floor_id,
                        wall_ids,
                        16*16*16*1)
  
  generate_model_images("suitcase",
                        env,
                        data_path,
                        output_data_path,
                        generate_data_path,
                        floor_id,
                        wall_ids,
                        16*16*16*2)


if __name__ == '__main__':
  main()
