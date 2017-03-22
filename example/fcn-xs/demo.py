# pylint: skip-file
import cv2
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import symbol_fcnxs
from collections import namedtuple
import mxnet as mx
print mx.__version__, cv2.__version__
Batch = namedtuple('Batch', ['data'])

def getpallete(num_cls):
    # this function is to get the colormap for visualizing the segmentation mask
    n = num_cls
    pallete = [0]*(n*3)
    for j in xrange(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete

pallete = getpallete(256)
img = "./data/street.jpg"
seg = img.replace("jpg", "png")
model_previx = "model/FCN8s_VGG16"
epoch = 19
ctx = mx.gpu(0)

def get_data(img_path, h, w):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    # resize to (h, w)
    img = cv2.resize(img, (w, h))
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    symbol = symbol_fcnxs.get_fcn8s_symbol(numclass=21, workspace_default=1536)
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    mod = mx.mod.Module(symbol, context=ctx)
    #data_shape = fcnxs_args["data"].shape
    data_shape = (1, 3, 800, 800)
    mod.bind(data_shapes=[('data', (1,3,data_shape[2], data_shape[3]))])
    mod.set_params(fcnxs_args, fcnxs_auxs)
    start = timer()
    data = get_data(img, data_shape[2], data_shape[3])
    mod.forward(Batch([mx.nd.array(data)]))
    output = mod.get_outputs()[0]
    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    time_elapsed = timer() - start
    out_img = Image.fromarray(out_img)
    out_img.putpalette(pallete)
    out_img.save(seg)
    print "Detection time: {:.6f} sec".format(time_elapsed)

if __name__ == "__main__":
    main()
