
import numpy as np

import sympy as sp
import sys
def rd(deg):
    return (deg / 180) * np.pi


# rotations around xyz axes
def rotx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -1 * np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def roty(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def rotxyz(theta_x, theta_y, theta_z):
    return np.matmul(np.matmul(rotx(theta_x), roty(theta_y)), rotz(theta_z))


def rotxzy(theta_x, theta_z, theta_y):
    return np.matmul(np.matmul(rotx(theta_x), rotz(theta_z)), roty(theta_y))


def rotyxz(theta_y, theta_x, theta_z):
    return np.matmul(np.matmul(roty(theta_y), rotx(theta_x)), rotz(theta_z))


def rotyzx(theta_y, theta_z, theta_x):
    return np.matmul(np.matmul(roty(theta_y), rotz(theta_z)), rotx(theta_x))


def rotzxy(theta_z, theta_x, theta_y):
    return np.matmul(np.matmul(rotz(theta_z), rotx(theta_x)), roty(theta_y))


def rotzyx(theta_z, theta_y, theta_x):
    return np.matmul(np.matmul(rotz(theta_z), roty(theta_y)), rotx(theta_x))


def rotxy(theta_x, theta_y):
    return np.matmul(rotx(theta_x), roty(theta_y))


def rotxz(theta_x, theta_z):
    return np.matmul(rotx(theta_x), rotz(theta_z))


def rotyx(theta_y, theta_x):
    return np.matmul(roty(theta_y), rotx(theta_x))


def rotyz(theta_y, theta_z):
    return np.matmul(roty(theta_y), rotz(theta_z))


def rotzx(theta_z, theta_x):
    return np.matmul(rotz(theta_z), rotx(theta_x))


def rotzy(theta_z, theta_y):
    return np.matmul(rotz(theta_z), roty(theta_y))


# skewsemetric representation
def skew(a):
    return np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])


# unskew matrix
def unskew(a):
    return np.array([[a[2, 1]], [a[0, 2]], [a[1, 0]]])


# quick column vector
def cv(a, b, c):
    return np.array([[a], [b], [c]])


# column vector to row vector
def uncv(a):
    size = len(a)
    b = np.zeros(size)
    size -= 1
    while size != -1:
        b[size] = a[size][0]
        size-=1

    return b


# rotation matrix with w_hat and theta
def rotwth(w_hat, theta):
    return np.identity(3) + (np.sin(theta) * skew(w_hat)) + (
            (1 - np.cos(theta)) * (np.matmul(skew(w_hat), skew(w_hat))))


# Logarithmic coordinates
def logcoord(R):
    theta = np.arccos(0.5 * (np.trace(R) - 1))
    what = np.array([[0], [0], [0]])
    if theta == 0:
        return theta, what
    else:
        if (np.trace(R) == -1):
            theta = np.pi
            guess1 = (1 / (np.sqrt(2 * (1 + R[2, 2])))) * np.array([[R[0, 2]], [R[1, 2]], [1 + R[2, 2]]])
            guess2 = (1 / (np.sqrt(2 * (1 + R[1, 1])))) * np.array([[R[0, 1]], [1 + R[1, 1]], [R[2, 1]]])
            guess3 = (1 / (np.sqrt(2 * (1 + R[0, 0])))) * np.array([[1 + R[0, 0]], [R[1, 0]], [R[2, 0]]])
            return 'Theta is pi and there are three guesses:: \nGuess 1 is:\n{}\nGuess 2 is:\n{}\nGuess 3 is:\n{}'.format(
                guess1, guess2, guess3)
        else:
            what = unskew((1 / (2 * np.sin(theta))) * (R - np.transpose(R)))
            return 'Theta is ::{}\nwhat is ::\n{}\nwhat*theta is ::\n{}'.format(theta, what, theta * what)


# Twist
def twist(wx, wy, wz, vx, vy, vz):
    return np.array([[wx], [wy], [wz], [vx], [vy], [vz]])


def twist_S_Thetadot(S, theta_dot):
    return S * theta_dot


# adj
def adj(T):
    adjt = np.zeros([6, 6])
    adjt[0:3, 0:3] = T[0:3, 0:3]
    adjt[3:6, 3:6] = T[0:3, 0:3]
    adjt[3:6, 0:3] = np.matmul(skew(cv(T[0, 3], T[1, 3], T[2, 3])), T[0:3, 0:3])
    return adjt


# Screw
def screwrot(q, shat, h):
    v = np.cross(-1 * shat, q) + h * shat
    s = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    s[0] = shat[0]
    s[1] = shat[1]
    s[2] = shat[2]
    s[3] = v[0]
    s[4] = v[1]
    s[5] = v[2]
    return s


# Magnitude
def mag(vector):
    return np.sqrt(sum(pow(element, 2) for element in vector))


# Twist to screw
def vtos(V):
    w = np.array([V[0], V[1], V[2]])
    s = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    if any([v != 0 for v in w]):
        shat = w / mag(w)
        sv = np.array([V[3], V[4], V[5]]) / mag(w)
        s[0] = shat[0]
        s[1] = shat[1]
        s[2] = shat[2]
        s[3] = sv[0]
        s[4] = sv[1]
        s[5] = sv[2]
        return s
    else:
        sv = np.array([V[3], V[4], V[5]]) / mag(V)
        s[3] = sv[0]
        s[4] = sv[1]
        s[5] = sv[2]
        return s


def inverse_trans(T):
    p = np.array([[T[0][3]], [T[1][3]], [T[2][3]]])
    R = np.array([[T[0][0], T[0][1], T[0][2]],
                  [T[1][0], T[1][1], T[1][2]],
                  [T[2][0], T[2][1], T[2][2]]])
    T_R = np.transpose(R)
    neg_RT_p = (-1) * (T_R @ p)
    inverted_trans = np.array([[T_R[0][0], T_R[0][1], T_R[0][2], neg_RT_p[0][0]],
                               [T_R[1][0], T_R[1][1], T_R[1][2], neg_RT_p[1][0]],
                               [T_R[2][0], T_R[2][1], T_R[2][2], neg_RT_p[2][0]],
                               [0, 0, 0, 1]])

    return inverted_trans


# Transform
def transform(r, p):
    return np.array([[r[0, 0], r[0, 1], r[0, 2], p[0]],
                     [r[1, 0], r[1, 1], r[1, 2], p[1]],
                     [r[2, 0], r[2, 1], r[2, 2], p[2]],
                     [0, 0, 0, 1]])


# exponential coordinates for transformation matrix
def exp(s, theta):
    what = np.array([[s[0]], [s[1]], [s[2]]])
    sv = np.array([[s[3]], [s[4]], [s[5]]])
    print(what, "what")
    print(theta, "theta")
    a = rotwth(what, theta)
    t = np.zeros([4, 4])
    t[0:3, 0:3] = a


    g = np.identity(3) * theta + (1 - np.cos(theta)) * skew(what) + (theta - np.sin(theta)) * np.matmul(skew(what),
                                                                                                        skew(what))
    gsv = uncv(np.matmul(g, sv))
    t[0:3, 3] = gsv
    t[-1, -1] = 1
    return t


# cotangent func
def cot(theta):
    return np.cos(theta) / np.sin(theta)


# matrix logarithm for RMB
def matLog(t):
    s = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    theta = 0
    p = np.array([0.0, 0.0, 0.0])
    p[0] = t[0, 3]
    p[1] = t[1, 3]
    p[2] = t[2, 3]
    r = np.zeros([3, 3])
    r = t[0:3, 0:3]
    rows = len(r)
    cols = len(r[0])
    flag = True
    for i in range(0, rows):
        for j in range(0, cols):
            if (i == j and r[i][j] != 1):
                flag = False
                break

            if (i != j and r[i][j] != 0):
                flag = False
                break
    if (flag):
        theta = mag(p)
        sv = p / mag(p)
        s[3, 0] = sv[0]
        s[4, 0] = sv[1]
        s[5, 0] = sv[2]
    else:
        if np.trace(r) == -1:
            theta = np.pi
            guess1 = (1 / (np.sqrt(2 * (1 + r[2, 2])))) * np.array([[r[0, 2]], [r[1, 2]], [1 + r[2, 2]]])
            guess2 = (1 / (np.sqrt(2 * (1 + r[1, 1])))) * np.array([[r[0, 1]], [1 + r[1, 1]], [r[2, 1]]])
            guess3 = (1 / (np.sqrt(2 * (1 + r[0, 0])))) * np.array([[1 + r[0, 0]], [r[1, 0]], [r[2, 0]]])
            return 'Theta is pi and there are three guesses for sw:: \nGuess 1 is:\n{}\nGuess 2 is:\n{}\nGuess 3 is:\n{}'.format(
                guess1, guess2, guess3)
        else:
            theta = np.arccos(.5 * (np.trace(r) - 1))
            swskew = np.zeros([3, 3])
            swskew = (1 / (2 * np.sin(theta))) * (r - np.transpose(r))
            sw = unskew(swskew)
            s[0:3, 0] = sw[0:3, 0]
            g = np.zeros([3, 3])
            g = ((1 / theta) * np.identity(3)) - (0.5 * swskew) + ((1 / theta) - (.5 * cot(0.5 * theta))) * np.matmul(
                swskew, swskew)
            sv = np.matmul(g, np.array([[p[0]], [p[1]], [p[2]]]))
            s[3:6, 0] = sv[0:3, 0]
    return theta, s


# joints with shat and q to screws
def jtos(j):
    numscrews = len(j[0])
    s = np.zeros([6, numscrews])
    for i in range(0, numscrews):
        s[0:3, i] = j[0:3, i]
        s[3:6, i] = np.cross(-1 * j[0:3, i], j[3:6, i])
    return s


# POE
def poe(s, m, thetas):
    #s = jtos(j)
    t = np.identity(4)
    for i in range(0, len(s[0])):
        t = np.matmul(t, exp(s[0:6, i], thetas[i]))
    return np.matmul(t, m)


# Jacobian
def bjac(s, thetas):
    jac = np.zeros([6, len(s[0])])
    tmat = np.identity(4)
    jac[0:6, 0] = s[0:6, 0]
    for i in range(0, len(s[0]) - 1):
        tmat = np.matmul(tmat, exp(s[0:6, i], thetas[i]))
        jac[0:6, i + 1] = np.matmul(adj(tmat), s[0:6, i + 1])
    return jac



