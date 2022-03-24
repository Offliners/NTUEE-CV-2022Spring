import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        img_ = image.copy()
        down_sample = 0.5
        for _ in range(self.num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(img_)
            for i in range(1, self.num_guassian_images_per_octave):
                gaussian_images_in_octave.append(cv2.GaussianBlur(img_, (0, 0), self.sigma**i, self.sigma**i))
            
            img_ = cv2.resize(gaussian_images_in_octave[-1], (int(img_.shape[1] * down_sample), int(img_.shape[0] * down_sample)), interpolation=cv2.INTER_NEAREST)
            gaussian_images.append(gaussian_images_in_octave)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            dog_images_in_octave = []
            for j in range(1, self.num_guassian_images_per_octave):
                tmp = cv2.subtract(gaussian_images[i][j], gaussian_images[i][j - 1])
                # tmp1 = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
                # cv2.imwrite(f'DoG_output/DoG{i+1}_{j}.png', tmp1)
                dog_images_in_octave.append(tmp)
        
            dog_images.append(dog_images_in_octave)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_octaves):
            for j in range(1, self.num_DoG_images_per_octave - 1):
                prev_img = dog_images[i][j - 1]
                middle_img = dog_images[i][j]
                next_img = dog_images[i][j + 1]

                for x in range(1, middle_img.shape[0] - 1):
                    for y in range(1, middle_img.shape[1] - 1):
                        if np.abs(middle_img[x][y]) <= self.threshold:
                            continue

                        prev_values = prev_img[x - 1: x + 2, y - 1: y + 2].flatten()
                        middle_values = middle_img[x - 1: x + 2, y - 1: y + 2].flatten()
                        next_values = next_img[x - 1: x + 2, y - 1: y + 2].flatten()
                        compare_values = np.concatenate((np.concatenate((prev_values, middle_values), axis=0), next_values), axis=0)
                        compare_values = np.delete(compare_values, 13)
                        if middle_img[x][y] <= np.min(compare_values) or middle_img[x][y] >= np.max(compare_values):
                            if i == 0:
                                keypoints.append([x, y])
                            else:
                                keypoints.append([int(x / down_sample), int(y / down_sample)])
                
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return np.array(keypoints)