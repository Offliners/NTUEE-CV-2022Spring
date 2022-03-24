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
        r = int((self.wndw_size - 1) / 2)
        LUT = np.exp(- np.linspace(0, 1, 256) ** 2 / (2 * self.sigma_r * self.sigma_r))
        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        s_kernel = np.exp(- (x * x + y * y) / (2 * self.sigma_s * self.sigma_s))
        
        output = np.zeros_like(img)
        for y in range(r, r + img.shape[0]):
            for x in range(r, r + img.shape[1]):
                if guidance.ndim == 3:
                    wgt = LUT[abs(padded_guidance[y - r:y + r + 1, x - r:x + r + 1, 0] - padded_guidance[y, x, 0])] * \
                        LUT[abs(padded_guidance[y - r:y + r + 1, x - r:x + r + 1, 1] - padded_guidance[y, x, 1])] * \
                        LUT[abs(padded_guidance[y - r:y + r + 1, x - r:x + r + 1, 2] - padded_guidance[y, x, 2])] * \
                        s_kernel
                else:
                    wgt = LUT[abs(padded_guidance[y - r:y + r + 1, x - r:x + r + 1] - padded_guidance[y, x])] * s_kernel
                
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * padded_img[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * padded_img[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * padded_img[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
        
        return np.clip(output, 0, 255).astype(np.uint8)