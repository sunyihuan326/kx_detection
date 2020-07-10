# -*- coding: utf-8 -*-
# @Time    : 2020/7/8
# @Author  : sunyihuan
# @File    : hot_cnn.py

import matplotlib.pyplot as plt

import numpy as np
import cv2

# The local path to our target image
img_path = './examples/cat_dog.png'

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)

# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = VGG16(weights=None)
model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
preds = preds.tolist()[0]
preds_3 =  sorted(preds, reverse=True)[:3]
print('preds_3: ', preds_3)
preds_3_index = [ preds.index(value) for value in preds_3]
index = np.argmax(preds)
print('index: ',index)

# 242, 243, 282
def cam(index, ):
    # This is the "african elephant" entry in the prediction vector
    african_elephant_output = model.output[:, index]
    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    return heatmap

heatmap = cam(index)
# print('preds_3_index : ',preds_3_index)
# for index in preds_3_index:
#     heatmap += cam(index)
heatmap /= np.max(heatmap)
# print(heatmap.shape)
# plt.matshow(heatmap)
# plt.show()

#
# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite('catdog_cam.jpg', superimposed_img)