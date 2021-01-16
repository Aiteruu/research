import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.imsave(sys.argv[1].split('.')[0] + '_depth.png', 1000/np.clip(np.squeeze(np.array(Image.open(sys.argv[1]), dtype='float32')) * 1000, 10, 1000))