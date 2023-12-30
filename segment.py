import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

leaf = cv2.imread('leaf1.JPG')
#image = tf.keras.preprocessing.image.load_img("leaf1.JPG", target_size=(128, 128))

#input_arr = tf.keras.preprocessing.image.img_to_array(image)
#input_arr = input_arr.astype('float32') / 255.
plt.imshow(leaf)
plt.show()
image

image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
image *= 1./255
plt.imshow(image)
plt.show()

hsv_leaf = cv2.cvtColor(np.float32(leaf), cv2.COLOR_RGB2HSV)
plt.imshow(hsv_leaf)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
pixel_colors = image.reshape((np.shape(leaf)[0]*np.shape(leaf)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

h, s, v = cv2.split(hsv_leaf)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")

plt.show()
h
s
v



from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
light_green_rgb = (125,159,100)
dark_green_rgb = (53,94,26)
dark_green = rgb_to_hsv(dark_green_rgb)
light_green = rgb_to_hsv(light_green_rgb)
lg_square = np.full((10, 10, 3), light_green, dtype=np.uint8)/255.0
dg_square = np.full((10, 10, 3), dark_green, dtype=np.uint8) /255.0
#lg_square = np.full((10, 10, 3), light_green, dtype=np.uint8) / 255.0
#dg_square = np.full((10, 10, 3), dark_green, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(dg_square)
plt.subplot(1, 2, 2)
plt.imshow(lg_square)
plt.show()



mask = cv2.inRange(image, light_green_rgb, dark_green_rgb)
result = cv2.bitwise_and(image, hsv_leaf, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
image


light_white = (0, 0, 200)
dark_white = (145, 60, 255)

mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()

final_mask = mask + mask_white

final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()