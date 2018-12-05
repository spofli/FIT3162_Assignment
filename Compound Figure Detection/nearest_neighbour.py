import numpy as np
import tensorflow as tf


def knn(x_vals_train, y_vals_train, x_vals_test, y_vals_test):
    # Setting up KNN
    sess = tf.Session()
    k = 5
    batch_size = 100

    # Placeholders
    _, feature_size = np.shape(x_vals_train)
    x_data_train = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
    y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    # Distance metric (L1)
    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

    # Get min distance index (Nearest neighbor)
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    top_k_yvals = tf.gather(y_target_train, top_k_indices)
    prediction = tf.sign(tf.reduce_sum(top_k_yvals, 1))

    # Calculate how many loops over training data
    num_loops = int(np.ceil(len(x_vals_test) / batch_size))

    evaluations = []
    # Training/Testing classifier
    for i in range(num_loops):
        min_index = i * batch_size
        max_index = min((i + 1) * batch_size, len(x_vals_train))
        x_batch = x_vals_test[min_index:max_index]

        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                                      y_target_train: np.transpose([y_vals_train])})
        predictions = np.array(np.reshape(predictions, (1, -1))[0]).astype(np.int)
        evaluations.extend(predictions)
    prediction_accuracy = np.mean(evaluations == y_vals_test)
    return prediction_accuracy
