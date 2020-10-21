import os
import time
import numpy as np
import pyopencl as cl

# visualization
import cv2
from matplotlib import pyplot as plt

from path import Path
from error_metrics import *

class ABS_DIFF():
    def __init__(self, kernel_source_path=None, VERBOSE=True):
        platform = cl.get_platforms()[0]    # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        if VERBOSE:
            print("platform:", platform)
            print("device:", device)
        self.ctx = cl.Context([device])      # Create a context with your device
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        src_path = os.path.join(os.getcwd(), 'ABS_DIFF.cl')
        if kernel_source_path is not None:
            src_path = kernel_source_path
        assert os.path.isfile(src_path), "File missing: {}".format(src_path)
        
        src_module = open(src_path, 'r')
        CL_SOURCE = src_module.read()
        prg = cl.Program(self.ctx, CL_SOURCE).build(options='-cl-mad-enable -cl-fast-relaxed-math')

        # kerner function
        self.get_cost = prg.get_cost
        self.wta = prg.wta2

    def get_costvolume(self, left_img, right_img, dispRange=64, thread_num=(16, 16, 4), host_mem=False):
        assert left_img.shape == right_img.shape, "Shape of both imgs are different"

        # image object configuration
        mf = cl.mem_flags
        if len(left_img.shape) == 3:
            assert left_img.shape[2] >= 3, "Unavailable type of imgs"
            # RGB/BGR
            if left_img.shape[2] == 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGBA)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGBA)
            fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
        elif len(left_img.shape) == 2:
            # GRAY
            fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        else:
            print("Unavailable type of imgs")
            exit()

        # arguments
        h = left_img.shape[0]
        w = left_img.shape[1]
        var_height = np.int32(h)
        var_width = np.int32(w)
        var_dispRange = np.int32(dispRange)

        if thread_num is not None:
            assert w % thread_num[0] == 0 and h % thread_num[1] == 0 and dispRange % thread_num[2] == 0
        
        # output
        costvolume = np.zeros((h, w, dispRange), dtype=np.float32) # main output

        #### numpy to cl::Image2D memory parsing
        ctx = self.ctx
        left_image2D = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(w, h), hostbuf=left_img)
        right_image2D = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(w, h), hostbuf=right_img)
        cost_buffer = cl.Buffer(ctx, mf.READ_WRITE, size=w*h*dispRange*4) # (w x h x dispRange) x float
        
        queue = self.queue
        #### get cost
        self.get_cost(
            queue,
            (w, h, dispRange), # global size
            (thread_num), # local size
            left_image2D, right_image2D, # input
            cost_buffer, # output
            var_width,
            var_dispRange
        ).wait()

        if host_mem:
            costvolume = np.zeros((h, w, dispRange), dtype=np.float32) # host memory
            cl.enqueue_copy(queue, costvolume, cost_buffer, is_blocking=True)
            return cost_buffer, costvolume
        else:
            return cost_buffer

    def get_depth(self, left_img, right_img, dispRange=64, thread_num=(16, 16, 4), wta_thread_num=(16, 16), host_mem=False):       # arguments
        # arguments
        h = left_img.shape[0]
        w = left_img.shape[1]
        var_width = np.int32(w)
        var_dispRange = np.int32(dispRange)

        if wta_thread_num is not None:
            assert w % wta_thread_num[0] == 0 and h % wta_thread_num[1] == 0

        # output
        result = np.zeros((h, w), dtype=np.uint8) # main output
        costvolume = np.zeros((h, w, dispRange), dtype=np.float32) # sub output

        # device memory -- output
        ctx = self.ctx
        mf = cl.mem_flags
        result_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, size=w*h) # cl.uchar(=np.uint8)

        # get cost_volume
        cost_results = self.get_costvolume(
            left_img,
            right_img,
            dispRange=dispRange,
            thread_num=thread_num,
            host_mem=host_mem
        )
        if host_mem:
            cost_buffer = cost_results[0]
            costvolume = cost_results[1]
        else:
            cost_buffer = cost_results

        queue = self.queue
        #### wta
        self.wta(
            queue,
            (w, h), # global size
            (wta_thread_num), # local size
            cost_buffer, # input
            result_buffer, # output
            var_width,
            var_dispRange
        ).wait()
        cl.enqueue_copy(queue, result, result_buffer, is_blocking=True)

        if host_mem:
            return result, costvolume
        else:
            return result

if __name__ == '__main__' :
    """
    Quarter-size (450 x 375) versions of our new data sets "Cones" and "Teddy" are available for download below.
    Each data set contains 9 color images (im0..im8) and 2 disparity maps (disp2 and disp6).
    The 9 color images form a multi-baseline stereo sequence,
    i.e., they are taken from equally-spaced viewpoints along the x-axis from left to right.
    The images are rectified so that all image motion is purely horizontal.
    To test a two-view stereo algorithm, the two reference views im2 (left) and im6 (right) should be used.
    Ground-truth disparites with quarter-pixel accuracy are provided for these two views.
    Disparities are encoded using a scale factor 4 for gray levels 1 .. 255, while gray level 0 means "unknown disparity".
    Therefore, the encoded disparity range is 0.25 .. 63.75 pixels. 
    """
    data_dir_path = Path('./middlebury')
    sub_dir_name = 'Cones' # [Aloe, Bowling, Cones, Midd, Reindeer, Teddy]
    prefix = data_dir_path/sub_dir_name

    # get imgs
    left_img_path = prefix/'{}Left.png'.format(sub_dir_name)
    left_img = cv2.imread(left_img_path).astype(np.float32)
    right_img_path = prefix/'{}Right.png'.format(sub_dir_name)
    right_img = cv2.imread(right_img_path).astype(np.float32)

    # get G.T
    gt_path = prefix/'gtLeft.png'
    gt = cv2.imread(gt_path, 0)

    # make zncc instance
    abs_diff = ABS_DIFF()

    # get depth
    dispRange = 64
    with_cost = False
    st_time = time.time()
    result = abs_diff.get_depth(
        left_img,
        right_img,
        dispRange=dispRange,
        thread_num=None,
        wta_thread_num=None,
        host_mem=with_cost)

    if with_cost:
        depth, cost = result
    else:
        depth = result

    # error
    print("BP error : {:.4f}".format(BP(depth[:, dispRange:], gt[:, dispRange:] / 4, 4.0)))  # BP 4pixel rate
    print("MSE : {:.4f}".format(MSE(depth[:, dispRange:], gt[:, dispRange:] / 4)))
    print("Elapsed : {:.3f}ms".format(1000 * (time.time() - st_time)))

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(cv2.cvtColor(left_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[1].imshow(cv2.cvtColor(right_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[2].imshow(depth, cmap='gray')
    axs[3].imshow(gt / 4, cmap='gray')
    #plt.show()

    if with_cost:
        fig2, axs2 = plt.subplots(8, 8)
        cnt = 0
        for row in range(8):
            for col in range(8):
                axs2[row, col].imshow(cost[:, :, cnt], cmap='gray')
                cnt += 1
    plt.show()

