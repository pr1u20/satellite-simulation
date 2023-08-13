import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
    
def transform_vector_world_to_satellite(vector_world, quaternion):

    """Convert vector from world reference frame to satellite reference frame. 

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


def solution_1(A, B, C, D):

    assert C == 0, "Condition not met"
    assert D == B - A, "Condition not met"

    # c_1 and c_2 cane be any real integer
    c_1 = 0
    c_2 = 0

    a = 2 * np.pi * c_1 + np.pi
    b = 2 * np.pi * c_2 + np.pi

    return a, b

def solution_2(A,B,C,D):

    assert B*C != 0, "Condition not met"
    assert D == -A - np.sqrt(B ** 2 - C ** 2), "Condition not met"

    # c_1 and c_2 cane be any real integer
    c_1 = 0
    c_2 = 0

    a = 2 * np.pi * c_1 + np.pi
    b = 2 * (np.arctan((B - np.sqrt(B ** 2 - C ** 2)) / C) + np.pi * c_2)

    return a, b

def solution_3(A,B,C,D):
    return

def solution_4(A,B,C,D):
    return

def solution_5(A,B,C,D):
    return

def solution_7(A, B, C, D):

    # Check conditions
    condition1 = A ** 2 + 2 * A * D - B ** 2 + C ** 2 + D ** 2 != 0
    condition2 = A ** 2 - B ** 2 + 2 * B * D - C ** 2 - D ** 2 != 0

    expression_numerator = (
        A * B * (
            A ** 4 * C ** 2 - A ** 4 * D ** 2 - 2 * A ** 3 * C ** 2 * D - 2 * A ** 3 * D ** 3 -
            2 * A ** 2 * B ** 2 * C ** 2 + 2 * A ** 2 * B ** 2 * D ** 2 -
            2 * A ** 2 * B * C ** 2 * D - 2 * A ** 2 * B * D ** 3 - 2 * A ** 2 * C ** 4 -
            2 * A ** 2 * C ** 2 * D ** 2 - 2 * A * C * D ** 2 *
            math.sqrt(
                -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
                B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
            ) -
            2 * B * C * D ** 2 *
            math.sqrt(
                -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
                B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
            ) -
            2 * A ** 2 * C * D *
            math.sqrt(
                -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
                B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
            ) +
            2 * B ** 2 * C * D *
            math.sqrt(
                -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
                B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
            ) -
            2 * A * C ** 3 *
            math.sqrt(
                -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
                B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
            ) -
            2 * B * C ** 3 *
            math.sqrt(
                -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
                B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
            ) +
            2 * A * B ** 2 * C ** 2 * D + 2 * A * B ** 2 * D ** 3 - 4 * A * B * C ** 4 -
            8 * A * B * C ** 2 * D ** 2 - 4 * A * B * D ** 4 + 2 * A * C ** 4 * D +
            4 * A * C ** 2 * D ** 3 + 2 * A * D ** 5 + B ** 4 * C ** 2 - B ** 4 * D ** 2 +
            2 * B ** 3 * C ** 2 * D + 2 * B ** 3 * D ** 3 - 2 * B ** 2 * C ** 4 -
            2 * B ** 2 * C ** 2 * D ** 2 - 2 * B * C ** 4 * D - 4 * B * C ** 2 * D ** 3 - 2 * B * D ** 5 +
            C ** 6 + 3 * C ** 4 * D ** 2 + 3 * C ** 2 * D ** 4 + D ** 6
        )
    )
    expression_denominator = A ** 2 + B ** 2 - 2 * B * D + C ** 2 + D ** 2
    condition3 = expression_numerator / expression_denominator != 0

    assert condition1, "Condition not met"
    assert condition2, "Condition not met"
    assert condition3, "Condition not met"

    # c_1 and c_2 cane be any real integer
    c_1 = 0
    c_2 = 0

    # Calculate values for a and b
    a_numerator = math.sqrt(
    -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
    B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
    ) + 2 * A * C
    a_denominator = A ** 2 + 2 * A * D - B ** 2 + C ** 2 + D ** 2
    a = 2 * (math.atan(a_numerator / a_denominator) + math.pi * c_1)

    b_numerator = -math.sqrt(
    -A ** 4 + 2 * A ** 2 * B ** 2 + 2 * A ** 2 * C ** 2 + 2 * A ** 2 * D ** 2 -
    B ** 4 + 2 * B ** 2 * C ** 2 + 2 * B ** 2 * D ** 2 - C ** 4 - 2 * C ** 2 * D ** 2 - D ** 4
    ) - 2 * B * C
    b_denominator = A ** 2 - B ** 2 + 2 * B * D - C ** 2 - D ** 2
    b = 2 * (math.atan(b_numerator / b_denominator) + math.pi * c_2)

    return a, b

def solution_12(A,B,C,D):

    # Check conditions
    condition1 = A * C != 0
    condition2 = A + B != 0
    condition3_numerator = (
    B * (A ** 4 * math.sqrt(B ** 2 - C ** 2) + A ** 4 * B - 4 * A ** 3 * B * math.sqrt(B ** 2 - C ** 2) -
        4 * A ** 3 * B ** 2 + 4 * A ** 3 * C ** 2 + 6 * A ** 2 * B ** 3 + 6 * A ** 2 * B ** 2 * math.sqrt(B ** 2 - C ** 2) -
        4 * A ** 2 * C ** 2 * math.sqrt(B ** 2 - C ** 2) - 4 * A ** 2 * B * C ** 2 - 4 * A * B ** 4 + 4 * A * B ** 2 * C ** 2 -
        4 * A * B ** 3 * math.sqrt(B ** 2 - C ** 2) + B ** 5 + B ** 4 * math.sqrt(B ** 2 - C ** 2))
    )
    condition3_denominator = 0  # Placeholder, will be corrected in the next step
    condition3_denominator = (
    condition3_denominator if condition3_numerator == 0 else
    B * (A ** 4 * math.sqrt(B ** 2 - C ** 2) + A ** 4 * B - 4 * A ** 3 * B * math.sqrt(B ** 2 - C ** 2) -
        4 * A ** 3 * B ** 2 + 4 * A ** 3 * C ** 2 + 6 * A ** 2 * B ** 3 + 6 * A ** 2 * B ** 2 * math.sqrt(B ** 2 - C ** 2) -
        4 * A ** 2 * C ** 2 * math.sqrt(B ** 2 - C ** 2) - 4 * A ** 2 * B * C ** 2 - 4 * A * B ** 4 + 4 * A * B ** 2 * C ** 2 -
        4 * A * B ** 3 * math.sqrt(B ** 2 - C ** 2) + B ** 5 + B ** 4 * math.sqrt(B ** 2 - C ** 2))
    )
    condition3 = condition3_numerator != 0 or condition3_denominator != 0

    condition4 = D == np.sqrt(B ** 2 - C ** 2) - A

    assert condition1
    assert condition2
    assert condition3
    assert condition4

    # c_1 and c_2 cane be any real integer
    c_1 = 0
    c_2 = 0

    # Calculate values for a and b
    a_numerator = A - math.sqrt(B ** 2 - C ** 2)
    a_denominator = C
    a = 2 * (math.atan(a_numerator / a_denominator) + math.pi * c_2)

    b_numerator = (-A * math.sqrt(B ** 2 - C ** 2) - A * B + B * math.sqrt(B ** 2 - C ** 2) + B ** 2)
    b_denominator = C * (A + B)
    b = 2 * (math.atan(b_numerator / b_denominator) + math.pi * c_1)
    return






        

if __name__ == "__main__":

    #test_rotation_implementation()

    import math

    A = 0.01  # Replace with your desired value for A
    B = 1  # Replace with your desired value for B
    C = 0.3  # Replace with your desired value for C
    D = np.sqrt(B ** 2 - C ** 2) - A  # Replace with your desired value for D

    a, b = solution_7(A, B, C, D)


    print("a:", a)
    print("b:", b)
    print("C: ", A*np.sin(a) + B*np.sin(b))
    print("D: ", A*np.cos(a) - B*np.cos(b))