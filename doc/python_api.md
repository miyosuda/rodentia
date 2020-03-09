# rodentia.Environment class API


## Environment

Environment class constructor.

    Environment(width,
                height,
                bg_color=[0.0, 0.0, 0.0])

| Argument | Description                                 |
|----------|---------------------------------------------|
| width    | Screen width                                |
| height   | Screen height                               |
| bg_color | Background color (RGB value with 0.0 ~ 1.0) |



## add_camera_view

Add camera view for additional screen rendering.

    add_camera_view(width,
                    height,
                    bg_color=[0.0, 0.0, 0.0],
                    near=0.05,
                    far=80.0,
                    focal_length=50.0,
                    shadow_buffer_width=0)

| Argument            | Description                                          |
|---------------------|------------------------------------------------------|
| width               | Screen width                                         |
| height              | Screen height                                        |
| bg_color            | Background color (RGB value with 0.0 ~ 1.0)          |
| near                | Distance to near clip plane                          |
| far                 | Distance to far clip plane                           | 
| focal_length        | Focal Length of the camera                           |
| shadow_buffer_width | shadow buffer width (If 0, calculated automatically) |

| Return value type | Return value description                       |
|-------------------|------------------------------------------------|
| Int               | camera id                                      |



## add_box

Add box object.

    add_box(texture_path,
            half_extent,
            pos,
            rot=0.0,
            mass=0.0,
            detect_collision=False)

| Argument         | Description                                    |
|------------------|------------------------------------------------|
| texture_path     | A String, the path for the texture (.png file) |
| half_extent      | (x,y,z) Half extent size of the object.        |
| pos              | (x,y,z) Center of the object.                  |
| rot              | A float value for head angle (rot_y) or list (rx,ry,rz,rw) as the rotation quaternion of the object (in radian)                       |
| mass             | Float, mass of the object. if mass == 0, the object is treated as static object, and if mass > 0, the object is physically simulated. |
| detect_collision | Boolean, whether the object is included for collision check result.  If this argument is `True`, object's id is included when the agenet collides with this object. |

| Return value type | Return value description                       |
|-------------------|------------------------------------------------|
| Int               | object id                                      |



## add_sphere

Add sphere object.

    add_sphere(texture_path,
               radius,
               pos,
               rot=0.0,
               mass=0.0,
               detect_collision=False)

| Argument         | Description                                    |
|------------------|------------------------------------------------|
| texture_path     | A String, the path for the texture (.png file) |
| radius           | Float, raius of the shpere.                    |
| pos              | (x,y,z) Center of the object.                  |
| rot              | A float value for head angle (rot_y) or list (rx,ry,rz,rw) as the rotation quaternion of the object (in radian)                       |
| mass             | Float, mass of the object. if mass == 0, the object is treated as static object, and if mass > 0, the object is physically simulated. |
| detect_collision | Boolean, whether the object is included for collision check result. If this argument is `True`, object's id is included when the agenet collides with this object. |

| Return value type | Return value description                       |
|-------------------|------------------------------------------------|
| Int               | object id                                      |



## add_model

Add model object with Wavefront `.obj` format.

`.obj` file should be accompanied with `.mtl` material file and `.png` texture file.

    add_model(path,
              scale,
              pos,
              rot=0.0,
              mass=0.0,
              detect_collision=False)

| Argument         | Description                                    |
|------------------|------------------------------------------------|
| path             | A String, the path for .obj file.              |
| scale            | (x,y,z) Scaling of the object.                 |
| pos              | (x,y,z) Center of the object.                  |
| rot              | A float value for head angle (rot_y) or list (rx,ry,rz,rw) as the rotation quaternion of the object (in radian) |
| mass             | Float, mass of the object. if mass == 0, the object is treated as static object, and if mass > 0, the object is physically simulated. |
| detect_collision | Boolean, whether the object is included for collision check result. If this argument is `True`, object's id is included when the agenet collides with this object. |

| Return value type | Return value description                       |
|-------------------|------------------------------------------------|
| Int               | object id                                      |



## locate_object

Locate object to given position and orientataion.

    locate_object(id,
                  pos,
                  rot=0.0)

| Argument         | Description                                    |
|------------------|------------------------------------------------|
| id               | Int, Object id                                 |
| pos              | (x,y,z) Center of the object.                  |
| rot              | A float value for head angle (rot_y) or list (rx,ry,rz,rw) as the rotation quaternion of the object (in radian) |



## locate_agent

Locate agent to given position and orientataion.

    locate_agent(pos,
                 rot_y=0.0)

| Argument         | Description                                      |
|------------------|--------------------------------------------------|
| pos              | (x,y,z) Center of the agent                      |
| rot_y            | A float value for head angle (rot_y) (in radian) |



## set_light

Set light parameters

    set_light(dir=[-0.5, -1.0, -0.4],
              color=[1.0, 1.0, 1.0],
              ambient_color=[0.4, 0.4, 0.4],
              shadow_rate=0.2)

| Argument         | Description                                      |
|------------------|--------------------------------------------------|
| dir              | Light direction                                  |
| color            | (r,g,b) Light color (0.0~1.0)                    |
| ambient_color    | (r,g,b) Ambient light color (0.0~1.0)            |
| shadow_rate      | Float, shadow color rate (0.0~1.0)               |



## step

Step environment.

    step(action,
         num_steps=1)

| Argument         | Description                                      |
|------------------|--------------------------------------------------|
| action           | Int array with 3 elements. [Turn left/right, Move left/right, Move forward/back] |
| num_steps        | Int, teration count                                                              |


Return value is a dictionary which contains the result of this step calculation.

| Return value key  | Return value description                                  |
|-------------------|-----------------------------------------------------------|
| "collided"        | Int list, ids of the objects that collided with the agent |
| "screen"          | numpy nd_array of [width * height * 3] (uint8)            |




## render

Capture screen with the additional camera view.

    render(camera_id,
           pos,
           rot)

| Argument         | Description                          |
|------------------|--------------------------------------|
| camera_id        | Int. Camera id                       |
| pos              | Camera position                      |
| rot              | Quaternion of the camera orientation |

Return value is a dictionary which contains the result of this step calculation.

| Return value key  | Return value description                                  |
|-------------------|-----------------------------------------------------------|
| "screen"          | numpy nd_array of [width * height * 3] (uint8)            |



## remove_obj

Remove object from the environment.

    remove_obj(id)

| Argument     | Description                 |
|--------------|-----------------------------|
| id           | Int, Object id              |



## get_obj_info

Get object's current status information.

    get_obj_info(id)

| Argument     | Description                 |
|--------------|-----------------------------|
| id           | Int, Object id              |

Return value is a dictionary which contains the object's current status information.

| Return value key  | Return value description            |
|-------------------|-------------------------------------|
| "pos"             | Position (nd_array)                 |
| "velocity"        | Velocity (nd_array)                 |
| "rot"             | Rotation quatenion_array (nd_array) |


## get_agent_info

Get agent's current status information.

    get_agent_info()

Return alue is a dictionary which contains the agent's current state info.

| Return value key  | Return value description            |
|-------------------|-------------------------------------|
| "pos"             | Position (nd_array)                 |
| "velocity"        | Velocity (nd_array)                 |
| "rot"             | Rotation quatenion_array (nd_array) |
| "rot_y"           | Y rotation (float)                  |



## replace_obj_texture

Replace object texture(s).

If object is consist of multiple meshes, textures of these meshes can be replaced by applying list of texture pathes.

    replace_obj_texture(id,
                        texture_path)

| Argument     | Description                                         |
|--------------|-----------------------------------------------------|
| id           | Int, Object id                                      |
| texture_path | A string or list of string. path of the texture(s)  |
