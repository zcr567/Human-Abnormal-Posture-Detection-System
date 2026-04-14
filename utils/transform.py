import math
import numpy as np


def scale(points: np.array, k):
    points[:, 0] *= k[0]
    points[:, 1] *= k[1]
    points[:, 2] *= k[2]


def sin(x):
    return math.sin(x * math.pi / 180)


def cos(x):
    return math.cos(x * math.pi / 180)


def rotate(points, angles):
    ax, ay, az = angles
    rx = np.array([[cos(ax), -sin(ax), 0],
                   [sin(ax), cos(ax), 0],
                   [0, 0, 1]])
    ry = np.array([[1, 0, 0],
                   [0, cos(ay), -sin(ay)],
                   [0, sin(ay), cos(ay)]])
    rz = np.array([[cos(az), 0, sin(az)],
                   [0, 1, 0],
                   [-sin(az), 0, cos(az)]])
    points[:] = points @ rx @ ry @ rz


def move(points, vec):
    points[:, 0] += vec[0]
    points[:, 1] += vec[1]
    points[:, 2] += vec[2]
