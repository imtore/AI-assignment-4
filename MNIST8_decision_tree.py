from graphviz import render
from sklearn.tree import export_graphviz
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics


mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists


# Defining and fitting a DecisionTreeClassifier instance
tree = DecisionTreeClassifier(max_depth=251)
tree.fit(mnist, labels)

test_data = pandas.read_csv("MNIST/test_data.csv").values
test_label = pandas.read_csv("MNIST/test_label.csv").values

prediction = tree.predict(test_data)
print(classification_report(test_label, prediction))
print("Accuracy of test:", metrics.accuracy_score(test_label, prediction))

train_p = tree.predict(mnist.values)
print("Accuracy of train:", metrics.accuracy_score(labels.values, train_p))


# Visualize Decision Tree

# Creates dot file named tree.dot
export_graphviz(
    tree,
    out_file="myTreeName.dot",
    feature_names=list(mnist.columns),
    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    filled=True,
    rounded=True)
