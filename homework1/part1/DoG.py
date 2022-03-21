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
            for i in range(self.num_guassian_images_per_octave):
                gaussian_images_in_octave.append(cv2.GaussianBlur(img_, (0, 0), self.sigma**(i+1), self.sigma**(i+1)))
            
            img_ = cv2.resize(img_, (int(img_.shape[1] * down_sample), int(img_.shape[0] * down_sample)), interpolation=cv2.INTER_NEAREST)
            gaussian_images.append(gaussian_images_in_octave)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for prev, next in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(cv2.subtract(next, prev))
        
            dog_images.append(dog_images_in_octave)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique


        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return np.array(keypoints)