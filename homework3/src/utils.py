import numpy as np
import cv2

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    u_x = np.transpose([u[:, 0]])
    u_y = np.transpose([u[:, 1]])
    v_x = np.transpose([v[:, 0]])
    v_y = np.transpose([v[:, 1]])

    A_u = np.hstack((u_x, u_y, np.ones((N, 1)), np.zeros((N, 3)), -1 * u_x * v_x, -1 * u_y * v_x, -1 * v_x))
    A_v = np.hstack((np.zeros((N, 3)), u_x, u_y, np.ones((N, 1)), -1 * u_x * v_y, -1 * u_y * v_y, -1 * v_y))
    A = np.vstack((A_u, A_v))

    # TODO: 2.solve H with A
    _, _, VT = np.linalg.svd(A)
    h = VT[-1,:] / VT[-1,-1]
    H = h.reshape(3, 3)

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    u_x, u_y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax)) # , sparse=False
    u_x = u_x.reshape(-1).astype(int)
    u_y = u_y.reshape(-1).astype(int)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    u = np.vstack((u_x, u_y, np.ones(len(u_x))))

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        H_inv = np.linalg.inv(H)
        v = H_inv @ u
        v = v / v[-1]

        v_x = v[0]
        v_y = v[1]
        # v_x = np.round(v[0]).astype(int)
        # v_y = np.round(v[1]).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.where((v_x >= 0) & (v_x < w_src - 1) & (v_y >= 0) & (v_y < h_src - 1))
        # mask = np.where((v_x >= 0) & (v_x < w_src) & (v_y >= 0) & (v_y < h_src))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        u_x, u_y = u_x[mask], u_y[mask]
        v_x, v_y = v_x[mask], v_y[mask]

        # TODO: 6. assign to destination image with proper masking
        dst[u_y, u_x] = bilinear(src, v_x, v_y)
        # dst[u_y, u_x] = src[v_y, v_x]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        v = H @ u
        v = v / v[-1]
        v_x = np.round(v[0]).astype(int)
        v_y = np.round(v[1]).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.where((v_x >= 0) & (v_x < w_dst) & (v_y >= 0) & (v_y < h_dst))

        # TODO: 5.filter the valid coordinates using previous obtained mask
        u_x, u_y = u_x[mask], u_y[mask]
        v_x, v_y = v_x[mask], v_y[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[v_y, v_x] = src[u_y, u_x]

    return dst

def bilinear(image, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    
    wa = np.repeat((y1 - y) * (x1 - x), 3).reshape((-1, 3))
    wb = np.repeat((x1 - x) * (y - y0), 3).reshape((-1, 3))
    wd = np.repeat((x - x0) * (y1 - y), 3).reshape((-1, 3))
    wc = np.repeat((x - x0) * (y - y0), 3).reshape((-1, 3))

    return wa * image[y0, x0] + wb * image[y1, x0] + wc * image[y1, x1] + wd * image[y0, x1]