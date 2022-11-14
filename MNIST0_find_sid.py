import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas

data_frame = pandas.pandas.read_csv("MNIST/train_data.csv")

number6 = data_frame.loc[[1201]]

pixels = []
for column in number6:
    pixels.append(number6[column].values[0])

pixels = np.array(pixels, dtype='uint8')

# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))

plt.title('my sid\'s least significant number')
plt.imshow(pixels, cmap='gray')
plt.show()
