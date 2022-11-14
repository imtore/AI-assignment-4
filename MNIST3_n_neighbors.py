import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


MIN = 1
MAX = 30

mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists

test_data = pandas.read_csv("MNIST/test_data.csv").values
test_label = pandas.read_csv("MNIST/test_label.csv").values

features = []
for column in mnist:
    features.append(mnist[column].values)

features = list(zip(*features))
train_label = labels['0'].to_numpy()

distance_accuracy = []
uniform_accuracy = []

for i in range(MIN, MAX):
    model = KNeighborsClassifier(n_neighbors=i, weights='distance')
    model.fit(features, train_label)
    prediction = model.predict(test_data)
    distance_accuracy.append(metrics.accuracy_score(test_label, prediction))

    model = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    model.fit(features, train_label)
    prediction = model.predict(test_data)
    uniform_accuracy.append(metrics.accuracy_score(test_label, prediction))


x = range(MIN, MAX)


if(max([max(distance_accuracy), max(uniform_accuracy)]) == max(distance_accuracy)):
    max_n = distance_accuracy.index(max(distance_accuracy))
    method = "distance"
else:
    max_n = uniform_accuracy.index(max(uniform_accuracy))
    method = "uniform"

print("max accuracy: ", max([max(distance_accuracy), max(
    uniform_accuracy)]), " best N: ", max_n, " method: ", method)


plt.plot(np.array(x), np.array(distance_accuracy), color='g', label='distance')
plt.plot(np.array(x), np.array(uniform_accuracy), color='b', label='uniform')
plt.xlabel('n_neighbors')
plt.ylabel('test accuracy')
plt.legend(loc='upper left')
plt.show()
