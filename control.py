import pybullet as p
import numpy as np

from environment import SatelliteEnv
from utils import transform_vector_world_to_satellite

class PID():
	def __init__(self,KP,KI,KD,target = 0):
		self.kp = KP
		self.ki = KI
		self.kd = KD		
		self.target = target
		self.error_last = 0
		self.integral_error = 0
		self.saturation_max = None
		self.saturation_min = None
                
	def compute(self,pos,dt):
		error = self.target - pos #compute the error
		derivative_error = (error - self.error_last) / dt #find the derivative of the error (how the error changes with time)
		self.integral_error += error * dt #error build up over time
		output = self.kp*error + self.ki*self.integral_error + self.kd*derivative_error 
		self.error_last = error
		if output > self.saturation_max and self.saturation_max is not None:
			output = self.saturation_max
		elif output < self.saturation_min and self.saturation_min is not None:
			output = self.saturation_min
		return output
        
	def setLims(self,min,max):
		self.saturation_max = max
		self.saturation_min = min
                
class Strategy_1D():
    def __init__(self):
        # Target is specified in the compute method.
        self.controller = PID(0.7, 0, 0.1, target = 0)
        self.controller.setLims(-1, 1)
        self.dt = 1/240
        #p.addUserDebugPoints([[self.target, 0, 1]], pointColorsRGB=[[1, 0, 0]], pointSize = 2, lifeTime=0)

    def compute(self, obs):

        current_position = obs[:3]
        current_orientation = obs[3:7]
        target_position = obs[7:10]

        self.controller.target = target_position[0]

        output = self.controller.compute(current_position[0], self.dt)
        print(output)

        action = self.translation_action(output)

        return action

    def translation_action(self, output):

        thruster_0_activated = output
        thruster_1_activated = - output
        #thruster_0_activated, thruster_1_activated = (1, 0)
        angle_y_0 = 0
        angle_z_0 = 0
        angle_y_1 = 0
        angle_z_1 = 0

        action = np.array([thruster_0_activated, thruster_1_activated, angle_y_0, angle_z_0, angle_y_1, angle_z_1])

        return action

class Strategy_2D():
    def __init__(self):
        # Target is specified in the compute method.
        self.controller_x = PID(0.7, 0.0, 0.2, target = 0)
        self.controller_y = PID(0.7, 0.0, 0.2, target = 0)
        self.controller_x.setLims(-1, 1)
        self.controller_y.setLims(-1, 1)
        self.dt = 1/240
        #p.addUserDebugPoints([[self.target, 0, 1]], pointColorsRGB=[[1, 0, 0]], pointSize = 2, lifeTime=0)

    def compute(self, obs):

        current_position = obs[:3]
        current_orientation = obs[3:7]
        target_position = obs[7:10]


        connecting_vector = target_position - current_position
        # Can also be interpreted as the error between current position and target in satellite reference frame.
        relative_connecting_vector = transform_vector_world_to_satellite(connecting_vector, current_orientation)

        error = relative_connecting_vector

        self.controller_x.target = error[0]
        self.controller_y.target = error[1]


        # We changed the reference frame to the satellites, so the current position is 0.
        relative_current_position = [0,0,0]

        output_x = self.controller_x.compute(relative_current_position[0], self.dt)
        output_y = self.controller_y.compute(relative_current_position[1], self.dt)
        output = output_x, output_y
        print(output)

        action = self.translation_action(output)

        return action

    def translation_action(self, output):

        output_x, output_y = output

        if abs(output_x) > abs(output_y):

            thruster_0_activated = output_x
            thruster_1_activated = - output_x
            #thruster_0_activated, thruster_1_activated = (1, 0)
            angle_y_0 = 0
            angle_z_0 = 0
            angle_y_1 = 0
            angle_z_1 = 0

        else:
            thruster_0_activated = 1
            thruster_1_activated = 1
            #thruster_0_activated, thruster_1_activated = (1, 0)
            angle_y_0 = output_y
            angle_z_0 = 0
            angle_y_1 = output_y
            angle_z_1 = 0

        action = np.array([thruster_0_activated, thruster_1_activated, angle_y_0, angle_z_0, angle_y_1, angle_z_1])

        return action
    

if __name__ == "__main__":
	
    env = SatelliteEnv()
    strategy = Strategy_2D()
    obs, info = env.reset()
    print(obs)


    done = False
    while not done:
        # Choose an actionenv
        #action = strategy(obs)
        #action = env.action_space.sample()
        action = strategy.compute(obs)

        #print(action)
        # Perform the chosen action in the environment
        obs, reward, done, _, info = env.step(action)
        #print(reward)


    env.close()
    env.plot_results()