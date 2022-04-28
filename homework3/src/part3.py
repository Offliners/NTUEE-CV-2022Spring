import numpy as np
import cv2
from utils import solve_homography, warping


if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    dst = np.zeros((h, w, c))
    

    # TODO: call solve_homography() & warping
    output3_1 = None
    output3_2 = None

    cv2.imwrite('output3_1.png', output3_1)
    cv2.imwrite('output3_2.png', output3_2)