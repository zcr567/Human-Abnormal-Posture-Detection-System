"""A tool converting depth image to .xyz format point cloud data."""
import math

import numpy as np

img_path = "00005.npy"  # 测试文件
output = "output.npy"
height_angle = 58
width_angle = 87

width_k = math.tan(height_angle * math.pi / 360)
height_k = math.tan(width_angle * math.pi / 360)

depth = np.load(img_path) / 1000
result = np.zeros((depth.shape[0] * depth.shape[1], 3))
pos = np.indices(depth.shape)
pos[0] -= depth.shape[0] // 2
pos[1] -= depth.shape[1] // 2

result[:, 0] = (pos[1] * depth * width_k / (depth.shape[0]/2)).flatten()
result[:, 1] = (pos[0] * depth * height_k / (depth.shape[1]/2)).flatten()
result[:, 2] = depth.flatten()

with open(output, "wb+") as f:
    np.save(f, result)
    f.close()

if __name__ == '__main__':
    with open("output.xyz", "r+") as f:  # 输出到output.xyz
        for l in result:
            for n in l:
                f.write(repr(float(n)) + ", ")
            f.write("\n")
