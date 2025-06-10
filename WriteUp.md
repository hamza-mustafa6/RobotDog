Downloaded Conda on linux in order to have multiple environments.
Downloded Isaac Sim and Lab on linux.
SSH'ed into the Linux computer using my Mac in order to access Isaac Lab, however still cannot see the GUI.Attempted to use IsaacSim WebRTC however couldn't get it to work, will revist later. 
Used XQuarts in order to see the GUI of IsaacLab. It is significantly slower with a bad framerate, but it works. https://www.youtube.com/watch?v=hdXDMIvQuTs
Ran IsaacLabs Cartpole RL code and got it to work.
Visualized the log information of the 150 cartpole epochs using tensorboard. 

NextUp: Familiarize with IsaacLab code and documentation, and implementing PD Control on Unitree Go2 Robot. 

The Euler angles are a way to represent rotation in 3d space. We have the 3 angles, which are about the x, y, and z axis, thus, Euler angles have 3 values. The thing with Euler angles is that when 2 axis align, you basically lose an axis of rotation because both axes are rotating in the same direction. This is called gimbal lock.

Quaternions are another way of representing rotation in 3d space using a 4d vector, and the 4 values are 3 axes defined by you, and the 4th value is a scalar element representing the angle on that defined axes.

Quaternions are better for robotics and movement because they avoid gimbal lock. 
