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
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        ### TODO ###
        padded_img_norm = padded_img/255.
        padded_guidance_norm = padded_guidance/255.

        hs = np.array([[np.exp(-((i-self.pad_w)**2 + (j-self.pad_w) ** 2) / (2 * (self.sigma_s ** 2))) for i in range(self.wndw_size)] for j in range(self.wndw_size)]).reshape(-1)
        hr = np.zeros((img.shape[0]*img.shape[1], self.wndw_size * self.wndw_size))
        u_padded_img = np.zeros((img.shape[0]*img.shape[1], self.wndw_size * self.wndw_size, img.shape[2]))
        
        for i in range(img.shape[0] * img.shape[1]):
            x = i // img.shape[1]
            y = i % img.shape[1]
            hr[i] = np.exp(-((padded_guidance_norm[x:x+self.wndw_size, y:y+self.wndw_size] - padded_guidance_norm[x+self.pad_w, y+self.pad_w]) ** 2).reshape(self.wndw_size * self.wndw_size, -1).sum(axis=-1) / (2 * (self.sigma_r ** 2)))
            u_padded_img[i] = padded_img_norm[x: x + self.wndw_size, y: y + self.wndw_size].reshape(-1, img.shape[2])

        hshr = hs * hr
        output = ((u_padded_img * (hshr).reshape((*hshr.shape, -1))).sum(axis=1) / (hshr).sum(axis=-1).reshape((hshr.shape[0], -1))).reshape(img.shape) * 255

        return np.clip(output, 0, 255).astype(np.uint8)