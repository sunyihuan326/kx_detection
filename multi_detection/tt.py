from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np
from skimage import util

image_path = "C:/Users/sunyihuan/Desktop/85_new_cam/20191216_062713704.jpg"
image = Image.open(image_path)
image = ImageEnhance.Contrast(image)  # 对比度增强
image = image.enhance(random.uniform(0.6, 1.2))  # 增强系数[0.6, 1.2]
print(np.array(image))
image = util.random_noise(np.array(image), mode="gaussian")  # 加入高斯噪声,输出值为[0,1],需乘以255
image = image * 255

print(np.array(image).astype(int))
