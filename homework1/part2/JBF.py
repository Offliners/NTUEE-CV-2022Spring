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
        output = np.zeros_like(img)
        r = int((self.wndw_size - 1) / 2)
        LUTr = np.exp(- np.linspace(0, 1, 256) ** 2 / (2 * self.sigma_r * self.sigma_r))
        x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        LUTs = np.exp(- (x * x + y * y) / (2 * self.sigma_s * self.sigma_s))

        for row in range(self.pad_w, padded_guidance.shape[0] - self.pad_w):
            for col in range(self.pad_w, padded_guidance.shape[0] - self.pad_w):
                if padded_guidance.ndim == 3:
                    Wsr = LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1), 0] - padded_guidance[row, col, 0])] * \
                          LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1), 1] - padded_guidance[row, col, 1])] * \
                          LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1), 2] - padded_guidance[row, col, 2])] * \
                          LUTs
                else:
                    Wsr = LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1)] - padded_guidance[row, col])] * LUTs

                sum_Wsr = np.sum(Wsr)
                output[row-self.pad_w, col-self.pad_w, 0] = np.sum(Wsr * padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 0]) / sum_Wsr
                output[row-self.pad_w, col-self.pad_w, 1] = np.sum(Wsr * padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 1]) / sum_Wsr
                output[row-self.pad_w, col-self.pad_w, 2] = np.sum(Wsr * padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 2]) / sum_Wsr
        
        return np.clip(output, 0, 255).astype(np.uint8)