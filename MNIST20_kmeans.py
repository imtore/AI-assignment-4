from sklearn.cluster import KMeans
import pandas
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt

mnist = pandas.read_csv("MNIST/train_data.csv")  # numpy of lists
labels = pandas.read_csv("MNIST/train_label.csv")  # numpy of lists


kmeans = KMeans(n_clusters=10)

kmeans.fit(mnist)

centroids = kmeans.cluster_centers_
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

C_norm = (centroids - centroids.min())/(centroids.max() - centroids.min())


pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
transformed = pandas.DataFrame(pca.fit_transform(C_norm))


transformed = transformed.values

print(transformed)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#f442b6', '#f48341', '#9141f4']
for i in range(0, 10):
    label = 'Class ' + str(i)
    plt.scatter(transformed[i][0], transformed[i]
                [1], label=label, c=colors[i])
plt.legend()
plt.show()
