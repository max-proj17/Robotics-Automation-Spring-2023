import math

import numpy as np



#
# def print_hi(name):
#
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
# if __name__ == '__main__':
#     print_hi('PyCharm')
import rotations as rotations

from invert import invt
import rotations


def addMats(A, B):
    C = A + B
    return C


print(5)

bruh = "bruh"
bart = "bart"

print("Hello {} {}".format(bruh, bart))

e = np.array([0, 0, 1])
B = np.zeros([4, 3])
I3 = np.identity(3)
C = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Vector e:\n", e)
print("Shape of vector:\n", np.shape(e))
print("\nMatrix C\n", C)

F = np.random.rand(3, 3)
E = np.matmul(C, F)

E = np.round(E, 1)

print("\nMatrix C\n", E)
print("\nMatrix C Transpose\n", np.transpose(E))
rsb = rotations.rotzxy(np.pi/3, np.pi/6, np.pi/2)
ras = np.transpose(rotations.rotxz(np.pi/4, np.pi/2))

print(np.transpose(np.matmul(ras, rsb)), "\n")

rsb = rotations.rotz(30 * (np.pi/180)) @ rotations.rotx(40 * (np.pi/180))
w_s = rotations.skew(np.array([[3], [3], [2]]))
dot_rsb = w_s @ rsb

w_b = rotations.unskew(np.transpose(rsb) @ dot_rsb)

print(w_b, "^^This is w_b \n")

rsb = rotations.rotz(45 * (np.pi/180)) @ rotations.rotx(60 * (np.pi/180)) @ rotations.roty(30 * (np.pi/180))

print(rotations.logcoord(rsb), "\n")

print(rotations.unskew(np.transpose(rsb) @ rotations.skew(np.array([[1],[2],[3]]))), "\n")

print(np.array([[0.267], [0.535], [0.802]]) * (45*(np.pi/180)), "\n")

print(rotations.rotwth(np.array([[0.267], [0.535], [0.802]]), (45 * (np.pi/180))), "\n")

g = rotations.logcoord(rotations.rotyzx((np.pi/2), np.pi, (np.pi/2)))

print(g, "\n")

print(np.array([[-0.57735027], [-0.57735027], [-0.57735027]]) * 2.094395, "\n^^New")

exp_coord = rotations.cv(1,2,1)
w_hat = exp_coord/(math.sqrt(sum(pow(element, 2)for element in exp_coord)))
theta = (exp_coord/w_hat)
print(theta, "(exp_coord/w_hat) = THETA\n")
print(w_hat, "WHAT\n")
print(rotations.rotwth(w_hat, theta))
print(rotations.screwrot(np.array([1, 1, 2]), np.array([0.577,0.577,0.577]), 10))
Rsb = rotations.roty((np.pi)/4)
vb = np.array([[1],[2],[1],[0],[0],[0]])
ps = np.array([-1,-2,0]) #dont make it a column vector
transformation = rotations.trans(Rsb, ps)
adjoint = rotations.adj(transformation)
vs = adjoint @ vb
print(adjoint)