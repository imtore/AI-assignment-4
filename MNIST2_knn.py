import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists

# features = []
# for column in mnist:
#     features.append(mnist[column].values)

# features = list(zip(*features))
# train_label = labels['0'].to_numpy()
# # unpacking

# model = KNeighborsClassifier(n_neighbors=3, weights='distance')

# model.fit(features, train_label)

# test_data = pandas.read_csv("MNIST/test_data.csv").values
# test_label = pandas.read_csv("MNIST/test_label.csv").values

# prediction = model.predict(test_data)
# print(classification_report(test_label, prediction))
# print("Accuracy of test:", metrics.accuracy_score(test_label, prediction))

# train_p = model.predict(mnist.values)
# print("Accuracy of train:", metrics.accuracy_score(labels.values, train_p))


x_train, x_test, y_train, y_test = train_test_split(
    mnist, labels, test_size=0.15, random_state=0)

label_test = y_test.values
data_test = x_test.values

features = []
for column in x_train:
    features.append(x_train[column].values)
features_train = list(zip(*features))

label_train = y_train['0'].to_numpy()

model = KNeighborsClassifier(n_neighbors=3)

model.fit(features_train, label_train)

data_eval = pandas.read_csv("MNIST/test_data.csv").values
label_eval = pandas.read_csv("MNIST/test_label.csv").values

prediction = model.predict(data_eval)
print(classification_report(label_eval, prediction))
print("Accuracy of test:", metrics.accuracy_score(label_eval, prediction))

train_pr = model.predict(data_test)
print("Accuracy of train:", metrics.accuracy_score(label_test, train_pr))


bottom = 1
top = 30

distance_accuracy = []
uniform_accuracy = []

for i in range(bottom, top):
    model = KNeighborsClassifier(n_neighbors=i, weights='distance')
    model.fit(features_train, label_train)
    prediction = model.predict(data_eval)
    distance_accuracy.append(metrics.accuracy_score(label_eval, prediction))

    model = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    model.fit(features_train, label_train)
    prediction = model.predict(data_eval)
    uniform_accuracy.append(metrics.accuracy_score(label_eval, prediction))

x = np.array(range(bottom, top))

print(len(uniform_accuracy))
max_d = max(distance_accuracy)
max_u = max(uniform_accuracy)

if(max([max_d, max_u]) == max(distance_accuracy)):
    max_n = distance_accuracy.index(max(distance_accuracy))
    method = "distance"
else:
    max_n = uniform_accuracy.index(max(uniform_accuracy))
    method = "uniform"

print("test max accuracy: ", max([max(distance_accuracy), max(
    uniform_accuracy)]), " best N: ", max_n, " method: ", method)


plt.plot(x, np.array(distance_accuracy), color='g', label='distance')
plt.plot(x, np.array(uniform_accuracy), color='b', label='uniform')
plt.xlabel('n_neighbors')
plt.ylabel('test accuracy')
plt.legend(loc='upper left')
plt.show()


distance_accuracy = []
uniform_accuracy = []

distance_error = []
uniform_error = []

for i in range(bottom, top):
    model = KNeighborsClassifier(n_neighbors=i, weights='distance')
    model.fit(features_train, label_train)
    prediction = model.predict(data_test)
    distance_accuracy.append(metrics.accuracy_score(label_test, prediction))
    distance_error.append(metrics.mean_absolute_error(label_test, prediction))

    model = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    model.fit(features_train, label_train)
    prediction = model.predict(data_test)
    uniform_accuracy.append(metrics.accuracy_score(label_test, prediction))
    uniform_error.append(metrics.mean_absolute_error(label_test, prediction))

x = np.array(range(bottom, top))

max_d = max(distance_accuracy)
max_u = max(uniform_accuracy)

if(max([max_d, max_u]) == max(distance_accuracy)):
    max_n = distance_accuracy.index(max(distance_accuracy))
    method = "distance"
else:
    max_n = uniform_accuracy.index(max(uniform_accuracy))
    method = "uniform"

print("train max accuracy: ", max([max(distance_accuracy), max(
    uniform_accuracy)]), " best N: ", max_n, " method: ", method)


plt.plot(x, np.array(distance_accuracy), color='g', label='distance')
plt.plot(x, np.array(uniform_accuracy), color='b', label='uniform')
plt.plot(x, np.array(distance_error), color='r', label='distance_error')
plt.plot(x, np.array(uniform_error), color='y', label='uniform_error')

plt.xlabel('n_neighbors')
plt.ylabel('train accuracy')
plt.legend(loc='upper left')
plt.show()
