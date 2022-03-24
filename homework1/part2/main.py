import numpy as np
import cv2
import argparse
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    RGB_weights_1 = np.array([[0.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0],
                              [0.1, 0.0, 0.9],
                              [0.1, 0.4, 0.5],
                              [0.8, 0.2, 0.0]
                            ])
                        
    RGB_weights_2 = np.array([[0.1, 0.0, 0.9],
                              [0.2, 0.0, 0.8],
                              [0.2, 0.8, 0.0],
                              [0.4, 0.0, 0.6],
                              [1.0, 0.0, 0.0]
                            ])

    if args.setting_path == '1':
        sigma_s = 2
        sigma_r = 0.1
        RGB_weights = RGB_weights_1
    elif args.setting_path == '2':
        sigma_s = 1
        sigma_r = 0.05
        RGB_weights = RGB_weights_2

    print('CV2 BGR2GRAY')
    cal_error(img_rgb, img_gray, sigma_s, sigma_r)
    for weight in RGB_weights:
        img_gray = weight[0] * img_rgb[:, :, 0] + weight[1] * img_rgb[:, :, 1] + weight[2] * img_rgb[:, :, 2]
        cv2.imwrite('filterd_gray.png', img_gray)

        print(f'(R, G, B) weight: ({weight[0]}, {weight[1]}, {weight[2]})')
        cal_error(img_rgb, img_gray, sigma_s, sigma_r)
    

def cal_error(img_rgb, img_gray, sigma_s, sigma_r):
    jbf = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = jbf.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = jbf.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)

    error = np.sum(np.abs(bf_out.astype('int32') - jbf_out.astype('int32')))
    print(f'Error: {error}')
    print()

    # jbf_out = cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('filterd_rgb.png', jbf_out)


if __name__ == '__main__':
    main()