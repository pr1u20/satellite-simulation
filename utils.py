import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
    
def transform_vector_world_to_satellite(vector_world, quaternion):

    """Convert vector from world reference frame to satellite reference frame. 
    vector_world can refer to the vector connecting the current position of the 
    satelite and the target position. The output would be that same vector in
    the satellite reference frame.

    :param vector_world: vector from the world reference frame
    :type vector_world: list
    :param quaternion: orientation of the satellite q (x, y, z, w).
    :type quaternion: list
    :return: transformed vector
    :rtype: list
    """

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)

    # Convert the vector to a column vector
    vector_world = np.array(vector_world).reshape(-1, 1)

    # Apply inverse rotation to the vector
    vector_satellite = np.dot(np.linalg.inv(rotation_matrix), vector_world)

    return vector_satellite.flatten()  # Return as a flattened array

def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to rotation matrix.

    :param quaternion: q (x, y, z, w)
    :type quaternion: list
    :return: R, rotation matrix
    :rtype: np.ndarray
    """

    x, y, z, w = quaternion  # Assuming quaternion as [a, b, c, d]

    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return R

def test_rotation_implementation():
    """
    Compare our coordinate transformation functions with the scipy methods.
    """

    q = [-0.348, -0.003, -0.890, 0.296]
    vector = [1, 2, 3]

    #SCIPY
    # initialize scipy rotation class with quaternions
    r = R.from_quat(q)
    # get the rotation matrix
    scipy_matrix = r.as_matrix()
    # transform vector by q
    new_vector_scipy = r.apply(vector, inverse = True)


    # OUR OWN
    # Convert quaternions to rotation matrix
    matrix = quaternion_to_rotation_matrix(q)
    # transform the vector by q
    new_vector = transform_vector_world_to_satellite(vector, q)

    print("\nScipy rotation matrix calculation: \n", scipy_matrix)
    print("\nScipy transformed vector: \n", new_vector_scipy)
    print("\nOur own rotation matrix calculation: \n", matrix)
    print("\nOur own transformed vector: \n", new_vector)


        

if __name__ == "__main__":

    test_rotation_implementation()