from sklearn.ensemble import RandomForestClassifier
import pandas
from sklearn.metrics import classification_report
from sklearn import metrics


mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists


# Defining and fitting a DecisionTreeClassifier instance
tree = RandomForestClassifier(n_estimators=300)
train_label = labels['0'].to_numpy()
tree.fit(mnist, train_label)

test_data = pandas.read_csv("MNIST/test_data.csv").values
test_label = pandas.read_csv("MNIST/test_label.csv").values

prediction = tree.predict(test_data)
print(classification_report(test_label, prediction))
print("Accuracy of test:", metrics.accuracy_score(test_label, prediction))

train_p = tree.predict(mnist.values)
print("Accuracy of train:", metrics.accuracy_score(labels.values, train_p))
