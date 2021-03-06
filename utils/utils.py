from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import tensorflow as tf


# Req. 2-2	세팅 값 저장
def save_config():
	pass


# Req. 4-1	이미지와 캡션 시각화
def visualize_img_caption(img_paths, captions):
	img_path = '.\\datasets\\images\\images\\' + img_paths
	ndarray = img.imread(img_path)

	plt.imshow(ndarray)
	plt.title(captions)
	plt.show()