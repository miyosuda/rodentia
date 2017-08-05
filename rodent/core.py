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
  def __init__(self, width, height):
    self.env = rodent_module.Env(width=width, height=height)

  def add_box(self, half_extent, pos, rot, detect_collision):
    """
    convert list arg to numpy ndarray.
    """
    return self.env.add_box(half_extent=to_nd_float_array(half_extent),
                            pos=to_nd_float_array(pos),
                            rot=rot,
                            detect_collision=detect_collision)

  def add_sphere(self, radius, pos, rot, detect_collision):
    """
    convert list arg to numpy ndarray.
    """    
    return self.env.add_sphere(radius=2.0,
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
    self.env.locate_agent(pos=to_nd_float_array([0.0, 1.0, 0.0]),
                          rot=0.0)

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
