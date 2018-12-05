from sklearn.svm import SVC

def svm(x_vals_train, y_vals_train, x_vals_test, y_vals_test):
    classifier = SVC(kernel="rbf", C=10000, gamma="scale")
    classifier.fit(x_vals_train, y_vals_train)
    return classifier.score(x_vals_test, y_vals_test)