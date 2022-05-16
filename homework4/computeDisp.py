import numpy as np
import cv2.ximgproc as xip
import cv2
import itertools

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    window_size = 9
    padding_size = int(window_size / 2)
    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    Il_padding = cv2.copyMakeBorder(Il_gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)
    Ir_padding = cv2.copyMakeBorder(Ir_gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)
    
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    Il_pattern = np.zeros((h, w, window_size * window_size, 1), dtype=np.bool)
    Ir_pattern = np.zeros((h, w, window_size * window_size, 1), dtype=np.bool)
    for i, j in itertools.product(range(h), range(w)):
        Il_pattern[i, j] = (Il_padding[i: i + window_size, j: j + window_size] < Il_padding[i + padding_size, j + padding_size]).reshape(-1, 1)
        Ir_pattern[i, j] = (Ir_padding[i: i + window_size, j: j + window_size] < Ir_padding[i + padding_size, j + padding_size]).reshape(-1, 1)

    Il_cost = np.ones((h, w, max_disp + 1), dtype=np.float32) * (window_size ** 2)
    Ir_cost = np.ones((h, w, max_disp + 1), dtype=np.float32) * (window_size ** 2)
    
    for i, j in itertools.product(range(h), range(w)):
        Il_cost[i, j, :j + 1] = np.logical_xor(Il_pattern[i, j], Ir_pattern[i, j::-1])[:max_disp+1].sum(-1).sum(-1)
        Ir_cost[i, j, :w - j] = np.logical_xor(Ir_pattern[i, j], Il_pattern[i, j:w:])[:max_disp+1].sum(-1).sum(-1)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for shift in range(max_disp + 1):
        Il_cost[:, :, shift] = xip.jointBilateralFilter(Il_gray, Il_cost[:, :, shift], -1, 3, 9)
        Ir_cost[:, :, shift] = xip.jointBilateralFilter(Ir_gray, Ir_cost[:, :, shift], -1, 3, 9)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    Il_labels = np.argmin(Il_cost, axis=-1)
    Ir_labels = np.argmin(Ir_cost, axis=-1)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    consistency = np.zeros((h, w))
    for i, j in itertools.product(range(h), range(w)):
        if Ir_labels[i, j - Il_labels[i, j]] == Il_labels[i, j]:
            consistency[i, j] = 1
    
    for i, j in itertools.product(range(h), range(w)):
        if consistency[i, j] == 0:
            Il_labels[i, j] = max_disp
            for k in range(j - 1, -1, -1):
                if consistency[i, k]:
                    Il_labels[i, j] = min(Il_labels[i, j], Il_labels[i, k])
                    break

            for k in range(j + 1, w):
                if consistency[i, k]:
                    Il_labels[i, j] = min(Il_labels[i, j], Il_labels[i, k])
                    break

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), Il_labels.astype(np.float32), 18, 1)

    return labels.astype(np.uint8)