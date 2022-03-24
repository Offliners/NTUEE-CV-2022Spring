import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        LUT = np.exp(- np.linspace(0, 1, 256) ** 2 / (2 * self.sigma_r * self.sigma_r))
        x, y = np.meshgrid(np.arange(-self.pad_w, self.pad_w + 1), np.arange(-self.pad_w, self.pad_w + 1))
        s_kernel = np.exp(- (x * x + y * y) / (2 * self.sigma_s * self.sigma_s))
        
        output = np.zeros_like(img)
        for i in range(self.pad_w, self.pad_w + img.shape[0]):
            for j in range(self.pad_w, self.pad_w + img.shape[1]):
                if guidance.ndim == 3:
                    weight = LUT[abs(padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1, 0] - padded_guidance[i, j, 0])] * \
                             LUT[abs(padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1, 1] - padded_guidance[i, j, 1])] * \
                             LUT[abs(padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1, 2] - padded_guidance[i, j, 2])] * \
                             s_kernel
                else:
                    weight = LUT[abs(padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1] - padded_guidance[i, j])] * s_kernel
                
                weight_acc = np.sum(weight)
                output[i - self.pad_w, j - self.pad_w, 0] = np.sum(weight * padded_img[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1, 0]) / weight_acc
                output[i - self.pad_w, j - self.pad_w, 1] = np.sum(weight * padded_img[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1, 1]) / weight_acc
                output[i - self.pad_w, j - self.pad_w, 2] = np.sum(weight * padded_img[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1, 2]) / weight_acc
        
        return np.clip(output, 0, 255).astype(np.uint8)