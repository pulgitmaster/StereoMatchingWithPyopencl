import numpy as np

# bad pixel
# BP% = 1/N * Sum(|pred_depth(x,y) - G.T_depth(x,y)| > sigma)
def BP(pred, gt, sigma):
    assert len(pred.shape) == 2, "[BadPixel rate]invalid predicted depth shape, must be [H, W]"
    assert pred.shape == gt.shape, "[BadPixel rate]shape unmatch problem"
    
    h, w = pred.shape
    valid_mask = gt != 0
    abs_diff = np.abs(pred[valid_mask] - gt[valid_mask])
    abs_diff[abs_diff <= sigma] = 0
    abs_diff_sum = np.sum(abs_diff!=0).astype(np.float)
    return abs_diff_sum / (h * w)

# mean square error
# MSE = 1/N * Sum( (pred_depth(x,y) - G.T_depth(x,y))^2 )
def MSE(pred, gt):
    assert len(pred.shape) == 2, "[MSE]invalid predicted depth shape, must be [H, W]"
    assert pred.shape == gt.shape, "[MSE]shape unmatch problem"
    
    h, w = pred.shape
    valid_mask = gt != 0
    square_diff = np.abs(pred[valid_mask] - gt[valid_mask])**2
    square_diff_sum = np.sum(square_diff).astype(np.float)
    return square_diff_sum / (h * w)   
