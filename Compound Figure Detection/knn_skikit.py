from sklearn.neighbors import KNeighborsClassifier

def knn(x_vals_train, y_vals_train, x_vals_test, y_vals_test):
    classifier = KNeighborsClassifier(n_neighbors=15, metric="manhattan")
    classifier.fit(x_vals_train, y_vals_train)
    return classifier.score(x_vals_test, y_vals_test)