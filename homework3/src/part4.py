import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    w = 0
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # for all images to be stitched:
    # Reference : https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w += im1.shape[1]       
  
        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        v_points = np.array([kp1[m.queryIdx].pt for m in good_matches])
        u_points = np.array([kp2[m.trainIdx].pt for m in good_matches])

        # TODO: 2. apply RANSAC to choose best H
        p = 0.99
        e = 0.7
        s = 4
        epoch = int(np.log(1 - p) / np.log(1 - (1 - e) ** s))
        threshold = 4
        inline_num_max = 0
        H_best = np.eye(3)
        for i in tqdm(range(epoch)):            
            idx = np.random.choice(range(len(u_points)), 4, replace=False)
            u_sample, v_sample = u_points[idx], v_points[idx] 

            H = solve_homography(u_sample, v_sample)

            U = np.vstack((u_points.T, np.ones((1, len(u_points)))))
            V = np.vstack((v_points.T, np.ones((1, len(v_points)))))
         
            V_estimate = H @ U
            V_estimate = V_estimate / V_estimate[-1]
            
            distance = np.linalg.norm((V_estimate - V)[:-1,:], ord=1, axis=0)
            inline_num = sum(distance < threshold)
            
            if inline_num > inline_num_max:
                inline_num_max = inline_num
                H_best = H

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ H_best

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], w, w + im2.shape[1], direction='b') 

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)