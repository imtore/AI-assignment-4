import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# from sklearn.model_selection import GridSearchCV


# min = 1
# max = 3
# mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
# labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists

# features = []
# for column in mnist:
#     features.append(mnist[column].values)

# features = list(zip(*features))
# train_label = labels['0'].to_numpy()

# grid_params = {"max_depth": range(1, 800)}
# dt = DecisionTreeClassifier()
# grid_search = GridSearchCV(dt, grid_params, verbose=2,
#                            n_jobs=-1, scoring='accuracy')
# grid_search.fit(features, train_label)


# plt.plot(np.array(x), np.array(y_uniform), color='g')
# plt.plot(np.array(x), np.array(y_distance), color='b')
# plt.show()

mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists


# Defining and fitting a DecisionTreeClassifier instance


test_data = pandas.read_csv("MNIST/test_data.csv").values
test_label = pandas.read_csv("MNIST/test_label.csv").values

accs = []
for i in range(1, 800):
    tree = DecisionTreeClassifier(max_depth=i)
    tree.fit(mnist, labels)

    prediction = tree.predict(test_data)
    acc = metrics.accuracy_score(test_label, prediction)
    accs.append(acc)

maximum_depth = accs.index(max(accs))
print("MAX: ", max(accs), ", depth: ", maximum_depth)
x = np.array(range(1, 800))
y = np.array(accs)


train_p = tree.predict(mnist.values)
print("Accuracy of train:", metrics.accuracy_score(labels.values, train_p))

plt.plot(x, y, color='g')
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.show()
