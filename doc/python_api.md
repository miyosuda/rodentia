# rodent.Environment class API

## Environment

Environment class constructor.

    Environment(width,
                height,
                bg_color=[0.0, 0.0, 0.0])

#### Argsuments:

- `width`: Screen width
- `height`: Screen height
- `bg_color`: Background color (RGB value with 0.0 ~ 1.0)



## add_box

Add box object.

    add_box(texture_path,
            half_extent,
            pos,
            rot=0.0,
            mass=0.0,
            detect_collision=False):

#### Argsuments:

- `texture_path`: A String, the path for the texture (.png file)
- `half_extent`: (x,y,z) Half extent size of the box.
- `pos`: (x,y,z) Center of the box.
- `rot`: (rx,ry,rz) for rotation of the box, or single float, head angle of the box (in radian)
- `mass`: Float, mass of the object. if mass == 0, the object is treated as static object, and if mass > 0, the object is physically simulated.
- `detect_collision`: Boolean, whether the object is included for collision check result. If this argument is `True`, object's id is included when the agenet collides with this object.

#### Returns:

Int, object id.


## add_sphere

Add sphere object.

    add_sphere(texture_path,
               radius,
               pos,
               rot=0.0,
               mass=0.0,
               detect_collision=False)

#### Arguments:

- `texture_path`: A String, the path for the texture (.png file)
- `radius`: Float, raius of the shpere.
- `pos`: (x,y,z) Center of the sphere.
- `rot`: (rx,ry,rz) for rotation of the sphere, or single float, head angle of the sphere (in radian)
- `mass`: Float, mass of the object. if mass == 0, the object is treated as static object, and if mass > 0, the object is physically simulated.
- `detect_collision`: Boolean, whether the object is included for collision check result. If this argument is `True`, object's id is included when the agenet collides with this object.

#### Returns:

Int, object id.



## add_model

Add model object with Wavefront `.obj` format.

`.obj` file should be accompanied with `.mtl` material file and `.png` texture file.

    add_model(path,
              scale,
              pos,
              rot=0.0,
              mass=0.0,
              detect_collision=False)

#### Arguments:

- `path`: A String, the path for .obj file.
- `scale`: (x,y,z) Scaling of the object.
- `pos`: (x,y,z) Center of the box.
- `rot`: (rx,ry,rz) for rotation of the object, or single float, head angle of the object (in radian)
- `mass`: Float, mass of the object. if mass == 0, the object is treated as static object, and if mass > 0, the object is physically simulated.
- `detect_collision`: Boolean, whether the object is included for collision check result. If this argument is `True`, object's id is included when the agenet collides with this object.

#### Returns:

Int, object id.



## locate_object

Locate object to given position and orientataion.

    locate_object(id,
                  pos,
                  rot=0.0)

#### Argsuments:

- `id`: Int, Object id
- `pos`: (x,y,z) Object's location.
- `rot`: (rx,ry,rz) for rotation of the object, or single float, head angle of the object (in radian)



## locate_agent

Locate agent to given position and orientataion.

    locate_agent(pos,
                 rot=0.0)

#### Argsuments:

- `pos`: (x,y,z) Agent's location.
- `rot`: Float, head angle of the agent (in radian)



## set_light

Set light parameters

    set_light(dir=[-0.5, -1.0, -0.4],
              color=[1.0, 1.0, 1.0],
              ambient_color=[0.4, 0.4, 0.4],
              shadow_rate=0.2)

#### Argsuments:

- `dir`: (x,y,z) Light direction.
- `color`: (r,g,b) Light color (0.0~1.0)
- `ambient_color`: (r,g,b) Ambient light color (0.0~1.0)
- `shadow_rate`: A float, shadow color rate (0.0~1.0)

## step

    step(action, num_steps=1)

#### Argsuments:

- action: Int array with 3 elements. [turn left/right, move left/right, move forward/back]
- num_steps: Int, teration count.

#### Returns:

A dictionary which contains the result of this step calculation.

- `"screen"`: numpy nd_array of [width * height * 3] (uint8)
- `"collided"` Int list, ids of the objects that collided with the agent.



## remove_obj

Remove object from the environment.

    remove_obj(id)

#### Argsuments:

- `id`: Int, Object id



## get_obj_info

Get object's current status information.

    get_obj_info(id)

#### Argsuments:

- `id`: Int, Object id

#### Returns:

A dictionary which contains the object's current status information.

- `"pos"`: numpy nd_array (float32)
- `"velocity"`: numpy nd_array (float32)
- `"euler_angles"`: numpy nd_array (float32)



## get_agent_info

Get agent's current status information.

    get_agent_info()

#### Returns:

A dictionary which contains the agent's current state info.

- `"pos"`: numpy nd_array (float32)
- `"velocity"`: numpy nd_array (float32)
- `"euler_angles"`: numpy nd_array (float32)
	


## replace_obj_texture

Replace object texture(s).

If object is consist of multiple meshes, textures of these meshes can be replaced by applying list of texture pathes.

    replace_obj_texture(id, texture_path)

#### Argsuments:

- `id`: Int, Object id
- `texture_path`: A string or list of string. path of the texture(s)
