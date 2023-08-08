import pybullet as p
import time
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class SatelliteEnv(gym.Env):
    
    def __init__(self):

        physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,0)
        planeId = p.loadURDF("plane.urdf") # creates the square plane in simulation.
        cubeStartPos = [0,0,1] # define the cube start position
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0]) # define start orientation
        self.boxId = p.loadURDF("cube.urdf",cubeStartPos, cubeStartOrientation) # load the object and set the pos and orientatiomn
        self.mass, _, self.CoG, *_ = p.getDynamicsInfo(self.boxId, -1) 
        self.F = 1000 # force applied by each thruster in Newtons

        self.fps = 24
        self.simulation_time = 10# max duration of the simulation before reset
        self._end_step = self.fps * self.simulation_time # last timestep of the simulation.
        self.reward = 0
        self.action_space = spaces.Dict(
            {
                "thruster_activation": spaces.Box(0, 1, shape=(2,), dtype=int),
                "vectoring": spaces.Box(-16, 16, shape=(4,), dtype=int),
            }
        )

        self.observation_space = spaces.Box(0, 1, shape=(2,))

        self.initialize()
    
    def initialize(self):
        # Function to be runned everytime the class is initialized and the environment is reset
        self.truncated = False
        self._done = False
        self._current_step = 0
    
    def reset(self, seed = None):
        # Reset the environment and return the initial observation
        self.initialize()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):

        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        
        return (cubePos, cubeOrn)
 
    def _get_info(self):
        
        return {}
    
    def step(self, action):

        thruster_0_activated, thruster_1_activated = action['thruster_activation']
        angle_y_0, angle_z_0, angle_y_1, angle_z_1 = action['vectoring']

        if thruster_0_activated == 1:
            self.apply_force(force = 10, y_angle = angle_y_0, z_angle = angle_z_0, thruster_number = 0)
        if thruster_1_activated == 1:
            self.apply_force(force = 10, y_angle = angle_y_1, z_angle = angle_z_1, thruster_number = 1)

        p.stepSimulation()

        time.sleep(1./self.fps)
        
        reward = 0
        self.reward += reward
            
        if self._current_step == self._end_step:
            self._done = True
            observation = self._get_obs()
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
            self.force_vector = (resultant_x, opposite_y, opposite_z)
            self.thruster_position = [-0.5,0,0] # Position of thruster with respect to satellite origin.

        elif thruster_number == 1:
            # Thruster is placed on opposite direction to thruster 1, so the sign of resultant_x is reversed
            self.force_vector = (-resultant_x, opposite_y, opposite_z)
            self.thruster_position = [0.5,0,0]
    
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
    
    def render(self, mode = 'human'):
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
        return
        
    
    def close(self):
        p.disconnect()
        

if __name__ == "__main__":

    env = SatelliteEnv()
    obs, info = env.reset()
    

    done = False
    while not done:
        # Choose an actionenv
        #action = strategy(obs)
        action = env.action_space.sample()
    
        # Perform the chosen action in the environment
        obs, reward, done, _, info = env.step(action)
    
    
    env.close()
