import pybullet as p
import time
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class SatelliteEnv(gym.Env):
    
    def __init__(self, render = True):

        self.render_GUI = render

        if self.render_GUI:
            physicsClient = p.connect(p.GUI)# p.GUI or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)# p.GUI or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,0)
        planeId = p.loadURDF("plane.urdf") # creates the square plane in simulation.
        self.start_position = [0,0,1] # define the cube start position
        self.target_position = [2,3,2]
        self.start_orientation = p.getQuaternionFromEuler([0,0,0]) # define start orientation
        self.boxId = p.loadURDF("cube.urdf",self.start_position, self.start_orientation) # load the object and set the pos and orientatiomn
        self.mass, _, self.CoG, *_ = p.getDynamicsInfo(self.boxId, -1) 

        self.F = 1 # force applied by each thruster in Newtons
        self.fps = 24
        self.simulation_time = 20 # max duration of the simulation before reset
        self._end_step = self.fps * self.simulation_time # last timestep of the simulation.
        self.dt = 1 / self.fps
        p.setTimeStep(self.dt) # If timestep is set to different than 240Hz, it affects the solver and other changes need to be made (check pybullet Documentation)
        self.realTime = True # If True simulation runs in real time. Change to False if training reinforcement learning.

        self.thruster_positions = np.array([[-0.5,0,0], [0.5,0,0]])

        """
        self.action_space = spaces.Dict(
            {
                "thruster_activation": spaces.Box(0, 1, shape=(2,), dtype=int),
                "vectoring": spaces.Box(-16, 16, shape=(4,), dtype=float),
            }
        )
        """
        self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,))

        self.initialize()
    
    def initialize(self):
        # Function to be runned everytime the class is initialized and the environment is reset
        self.truncated = False
        self._done = False
        self._current_step = 0
        self.reward = 0
        self.target_position = [2,3,2]
        self.previous_position = self.start_position
        self.current_position = self.start_position
        self.current_orientation = self.start_orientation
        p.resetBasePositionAndOrientation(self.boxId, self.current_position, self.current_orientation)
        self.actual_positions = []
        self.target_positions = []

        if self.render_GUI:
            p.addUserDebugLine(self.current_position, self.target_position, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
    
    def reset(self, seed = None):
        # Reset the environment and return the initial observation
        #p.resetSimulation()
        self.initialize()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):

        self.current_position, current_orientation = p.getBasePositionAndOrientation(self.boxId)
        # Data type of Spaces.Box is float32.
        obs = np.concatenate((self.current_position, current_orientation, self.target_position)).astype(np.float32)
        
        return obs
 
    def _get_info(self):

        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        distance_from_target = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.current_position))**2))
        
        return {'Current Position (m)': cubePos, "Target Position (m)": self.target_position, "Distance from target": distance_from_target, "Rewards": self.reward}
    
    def step(self, action):

        thruster_0_activated, thruster_1_activated = action[:2]
        #thruster_0_activated, thruster_1_activated = (1, 0)
        angle_y_0, angle_z_0, angle_y_1, angle_z_1 = action[2:]*16
        #angle_y_0, angle_z_0, angle_y_1, angle_z_1 = (15, 0, 15, 0)

        self.current_position, _ = p.getBasePositionAndOrientation(self.boxId)

        # If the output signal from the PID is very small, smaller than 0.01 (for example), the thruster is not activated. This reduces fuel usage, but decreases accuracy.
        if thruster_0_activated > 0.01:
            self.apply_force(force = self.F, y_angle = angle_y_0, z_angle = angle_z_0, thruster_number = 0)
        if thruster_1_activated > 0.01:
            self.apply_force(force = self.F, y_angle = angle_y_1, z_angle = angle_z_1, thruster_number = 1)

        p.stepSimulation()

        if self.realTime:
            time.sleep(1./self.fps)

        # Save the data from the simulation. Positions, target positions, orientations
        self.data_recorder()
        
        reward = self.reward_calculation()
        self.reward += reward

        self.previous_position = self.current_position
            
        if self._current_step == self._end_step:
            self._done = True
            observation = self._get_obs()
            print(observation)
            info = self._get_info()
            print(info)
        
        else:
            self._current_step += 1
                
            observation = self._get_obs()
            info = self._get_info()    
        return observation, reward, self._done, self.truncated, info
    
    def get_force_vector(self, force, y_angle, z_angle, thruster_number = 0):
        """Given the thruster angles find and force, find the force vector.

        :param force: Force applied by the thruster
        :type force: float
        :param y_angle: the angle in degrees that the thruster is vectored towards y-axis.
        :type y_angle: float
        :param z_angle: the angle in degrees that the thruster is vectored towards z-axis. 
        :type z_angle: float
        :param thruster_number: The number of the thruster you want to activate, defaults to 1
        :type thruster_number: int, optional
        """

        y_angle = y_angle * np.pi / 180
        z_angle = z_angle * np.pi / 180
        opposite_y = force * np.sin(y_angle)
        opposite_z = force * np.sin(z_angle)
        opposite_total = np.sqrt(opposite_y**2 + opposite_z**2)
        resultant_x = np.sqrt(force**2 - opposite_total**2)

        force_magnitude_check = np.sqrt(resultant_x**2 + opposite_y**2 + opposite_z**2)

        assert round(force_magnitude_check, 5) == force, f"Component forces {force_magnitude_check} do not equate to initial force {force}."

        # In the x_direction the resultant_x should have different sign. Unless you don't want one of the thrusters to activate.
        # In the y_direction the opposite_y should have opposite signs for rotation and equal sign for translation.
        # In the z_direction the opposite_z should have opposite signs for rotation and equal sign for translation.

        if thruster_number == 0:
            self.force_vector = np.array([resultant_x, opposite_y, opposite_z])

        elif thruster_number == 1:
            # Thruster is placed on opposite direction to thruster 1, so the sign of resultant_x is reversed
            self.force_vector = np.array([-resultant_x, opposite_y, opposite_z])

        self.thruster_position = self.thruster_positions[thruster_number] # Position of thruster with respect to satellite origin.
    
    def apply_force(self, force: float, y_angle: float, z_angle: float, thruster_number: int = 0) -> None:
        """Function to apply force by the thruster in a given timestep

        :param force: Force applied by the thruster
        :type force: float
        :param y_angle: the angle in degrees that the thruster is vectored towards y-axis.
        :type y_angle: float
        :param z_angle: the angle in degrees that the thruster is vectored towards z-axis. 
        :type z_angle: float
        :param thruster_number: The number of the thruster you want to activate, defaults to 1
        :type thruster_number: int, optional
        """

        self.get_force_vector(force, y_angle, z_angle, thruster_number)

        p.applyExternalForce(self.boxId, -1, self.force_vector, self.thruster_position, p.LINK_FRAME) # in Newton # WORLD_FRAME p.LINK_FRAME
        #p.applyExternalForce(self.boxId, -1, (-resultant_x, -opposite_y, -opposite_z), [-0.5,0,0], p.LINK_FRAME)

        self.draw_thrust_lines()

    def reward_calculation(self):

        distance_from_target_prev = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.previous_position))**2))
        distance_from_target = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.current_position))**2))

        #reward =  50000*(distance_from_target_prev - distance_from_target) / distance_from_target

        
        if distance_from_target <= distance_from_target_prev:
            reward = 1 / distance_from_target
        else:
            reward = 0
        

        return reward

    def draw_thrust_lines(self):
        if self.render_GUI:
            line_lenght = 0.1
            p.addUserDebugLine(self.thruster_position, self.thruster_position - line_lenght*self.force_vector, lineColorRGB=[1.00,0.25,0.10], lineWidth=4.0, lifeTime=self.dt, parentObjectUniqueId = self.boxId)
    
    def data_recorder(self):
        self.actual_positions.append(self.current_position)
        self.target_positions.append(self.target_position)

    def render(self, mode = 'human'):
        self.render_GUI = True
        return
        
    def plot_training(self):
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Rewards")
        
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_balances)
        plt.title("Balances (£)")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_trades)
        plt.title("Number of trades")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_success)
        plt.title("Successful trades (%)")
        
        plt.show()
        
    
    def plot_results(self, name = "performance_report", open_tab = False):

        self.actual_positions = np.array(self.actual_positions)
        self.target_positions = np.array(self.target_positions)


        plt.subplot(2, 2, 1)
        plt.plot(self.actual_positions[:, 0],  label='Actual')
        plt.plot(self.target_positions[:, 0], label='Target')
        plt.title("X positions")

        plt.subplot(2, 2, 2)
        plt.plot(self.actual_positions[:, 1],  label='Actual')
        plt.plot(self.target_positions[:, 1], label='Target')
        plt.title("Y positions")

        plt.subplot(2, 2, 3)
        plt.plot(self.actual_positions[:, 2],  label='Actual')
        plt.plot(self.target_positions[:, 2], label='Target')
        plt.title("Z positions")

        plt.tight_layout()
        plt.show()
        return
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":

    env = SatelliteEnv()
    obs, info = env.reset()
    print(obs)
    

    done = False
    while not done:
        # Choose an actionenv
        #action = strategy(obs)
        action = env.action_space.sample()

        #print(action)
        # Perform the chosen action in the environment
        obs, reward, done, _, info = env.step(action)
        #print(reward)
    
    
    env.close()
    env.plot_results()