import cv2
import numpy as np
from os import walk


def main():
    directory = "../data_gatherer/depth_data/0"
    savenum = 1
    for f in list(walk(directory))[0][2]:
        path = directory + "/" + f
        while True:
            img = np.load(path)

            # for depth only
            img = img.astype(np.float32)
            img *= (255 / 10000)
            img = 255 - img
            img = img.astype(np.uint8)
            img = np.dstack((img, img, img))

            cv2.imshow('Image', img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('g'):
                cv2.destroyAllWindows()
                cv2.imwrite(f"{savenum:03}.png", img)
                savenum += 1
                break
            elif key & 0xFF == ord('f'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main()
