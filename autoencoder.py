# Typical setup to include TensorFlow.
import tensorflow as tf
import glob
import os
import numpy as np
import scipy.ndimage as image

place = os.getcwd()
dataPath = place+"/data/small_pics/"

data = glob.glob(os.path.join(dataPath, "*.jpg"))
images = [image.imread(path, mode="RGB").astype(np.float) for path in data]

print(len(images),images[0].shape)
