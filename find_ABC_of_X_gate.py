######################################################################

# Lapu Matthias 

# The goal of this script is to find the values of alpha, theta0, theta1 
# and theta2. That will give us the unitary matrix X.
# We will use the formula : 
# U = exp(i*alpha) * Rz(theta2) * Ry(theta1) * Rz(theta0)
#
# We will brute force the values. (Yes, It's not elegant, but I find
# it faster than trying to solve the equation with a pen and paper.)

######################################################################

from math import *
import cmath
import numpy as np

# The First question asks us to find an ABC decomposition to find the unitary 
# matrix X. 

## It's always nice to have a function that converts the values to pi.
# it's just pretty printing.
def convert_to_pi(value):
    if np.isclose(value, 0):
        return "0"
    elif np.isclose(value, np.pi/4):
        return "pi/4"
    elif np.isclose(value, np.pi/2):
        return "pi/2"
    elif np.isclose(value, np.pi):
        return "pi"
    elif np.isclose(value, -np.pi/4):
        return "-pi/4"
    elif np.isclose(value, -np.pi/2):
        return "-pi/2"
    elif np.isclose(value, -np.pi):
        return "-pi"
    else:
        return str(value)
        

## We define the Ry and Rz matrices as functions of the angle theta.
# 
def Ry_matrix(theta): 
    return np.array([[cos(theta/2), -sin(theta/2)], 
                     [sin(theta/2), cos(theta/2)]])

def Rz_matrix(theta):
    return np.array([[cmath.exp(-1j*theta/2), 0], 
                     [0, cmath.exp(1j*theta/2)]])

## To find the unitary matrix X, 
# we use the formula : U = exp(i*alpha) * Rz(theta2) * Ry(theta1) * Rz(theta0)
def U(alpha,theta2,theta1,theta0):
    rz_part_1 = Rz_matrix(theta2)
    ry_part = Ry_matrix(theta1)
    rz_part_2 = Rz_matrix(theta0)
    return cmath.exp(1j*alpha) * np.dot(rz_part_1, np.dot(ry_part, rz_part_2))

## We can now find the values of alpha, theta0, theta1 and theta2
# that will give us the X matrix. Yes, we will brute force it.
# Yes it's O(n^4), but there's only 11 values to check.
def find_X_matrix():
    range_values = [0, pi/2, pi, pi/3, pi/4, pi/6, 
                    -pi/2, -pi, -pi/3, -pi/4, -pi/6]
    result = []
    for i in range_values: #alpha
        for j in range_values: #theta2
            for k in range_values: #theta1
                for l in range_values: #theta0
                    if np.allclose(U(i,j,k,l), np.array([[0, 1], [1, 0]])):
                        result.append((i,j,k,l))
    return result



res = find_X_matrix()

print ("The values of alpha, theta2, theta1 and theta0 are :")
for val in res : 
    print("----------------------------------")
    print(convert_to_pi(val[0]), convert_to_pi(val[1]), 
          convert_to_pi(val[2]), convert_to_pi(val[3]))    

print("----------------------------------")
print("Number of solutions found : ", len(res))
print("----------------------------------")
print("Number of iterations : ", 11**4)