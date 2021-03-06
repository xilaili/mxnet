# pylint: skip-file
import cv2
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import mxnet as mx

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

def get_data(img):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    #img = Image.open(img_path)
    #img = np.array(img, dtype=np.float32)
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    video = cv2.VideoCapture(0)

    count = 0
    while True:
        _, img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        start = timer()
        fcnxs_args["data"] = mx.nd.array(get_data(img), ctx)
        data_shape = fcnxs_args["data"].shape
        label_shape = (1, data_shape[2]*data_shape[3])
        fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
        exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_args)
        exector.forward(is_train=False)
        output = exector.outputs[0]
        out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
        time_elapsed = timer() - start
        out_img = Image.fromarray(out_img)
        out_img.putpalette(pallete)
        fname = 'data/'+str(count).zfill(7)+'.png'
        out_img.save(fname)
        count += 1
        #out_img.save(seg)
        print "Detection time: {:.6f} sec".format(time_elapsed)

if __name__ == "__main__":
    main()
