import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

mass, _, CoG, *_ = p.getDynamicsInfo(boxId, -1)
print(mass)


for i in range (10000):
    p.stepSimulation()
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    p.applyExternalForce(boxId, -1, (0, 10 , 0), -np.array(CoG), p.LINK_FRAME) # in Newton # WORLD_FRAME p.LINK_FRAME
    time.sleep(1./240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
