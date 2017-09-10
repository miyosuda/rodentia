import rodent_module
import numpy as np

def to_nd_float_array(list_obj):
  if isinstance(list_obj, np.ndarray):
    return list_obj
  else:
    return np.array(list_obj, dtype=np.float32)

def to_nd_int_array(list_obj):
  if isinstance(list_obj, np.ndarray):
    return list_obj
  else:
    return np.array(list_obj, dtype=np.int32)


class Environment(object):
  """
  rodent module Env wrapper.
  """
  def __init__(self, width, height, bg_color=[0.0, 0.0, 0.0]):
    self.env = rodent_module.Env(width=width,
                                 height=height,
                                 bg_color=to_nd_float_array(bg_color))

  def add_box(self, texture_path, half_extent, pos, rot, detect_collision):
    """
    convert list arg to numpy ndarray.
    """
    return self.env.add_box(texture_path=texture_path,
                            half_extent=to_nd_float_array(half_extent),
                            pos=to_nd_float_array(pos),
                            rot=rot,
                            detect_collision=detect_collision)

  def add_sphere(self, texture_path, radius, pos, rot, detect_collision):
    """
    convert list arg to numpy ndarray.
    """    
    return self.env.add_sphere(texture_path=texture_path,
                               radius=radius,
                               pos=to_nd_float_array(pos),
                               rot=rot,
                               detect_collision=detect_collision)

  def add_model(self, path, scale, pos, rot, detect_collision):
    """
    convert list arg to numpy ndarray.
    """
    return self.env.add_model(path=path,
                              scale=to_nd_float_array(scale),
                              pos=to_nd_float_array(pos),
                              rot=rot,
                              detect_collision=detect_collision)
  
  def locate_agent(self, pos, rot):
    """
    convert list arg to numpy ndarray.
    """
    self.env.locate_agent(pos=to_nd_float_array(pos),
                          rot=rot)

  def set_light_dir(self, dir):
    """
    convert list arg to numpy ndarray.
    """
    self.env.set_light_dir(dir=to_nd_float_array(dir))

  def step(self, action, num_steps=1):
    """
    convert list arg to numpy ndarray.
    """
    return self.env.step(action=to_nd_int_array(action), num_steps=num_steps)

  def remove_obj(self, id):
    """
    Remove object
    """
    return self.env.remove_obj(id=id)
