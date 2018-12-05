import numpy as np
import tensorflow as tf


def svm(x_vals_train, y_vals_train, x_vals_test, y_vals_test):
    # Create new SVM model
    sess = tf.Session()

    # Declare batch size, placeholders, and b-value
    batch_size = len(x_vals_train)
    _, feature_size = np.shape(x_vals_train)
    x_data = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    predict_data = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
    b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

    # Create gaussian kernel
    gamma = tf.constant(-np.float32(1 / (2 * (np.std(x_vals_train) ** 2))))
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),
                      tf.transpose(dist))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM model
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
    loss = tf.negative(tf.subtract(first_term, second_term))

    # Creating prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(predict_data), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(predict_data)))),
                          tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
    prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
    prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

    # gradient descent optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # initializing variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training classifier
    for step in range(100):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        batch_accuracy = []
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        acc_temp = sess.run(accuracy,
                            feed_dict={x_data: rand_x, y_target: rand_y, predict_data: rand_x})
        batch_accuracy.append(acc_temp)
        #if step % 25 == 0 or step == 99:
        #    print("Step: %d Loss: %d Batch accuracy: %.2f" % (step, temp_loss, np.mean(batch_accuracy)), end=', ', flush=True)
    # Evaluating classifier with unseen set
    [evaluations] = sess.run(prediction, feed_dict={x_data: x_vals_train,
                                                    y_target: np.transpose([y_vals_train]),
                                                    predict_data: x_vals_test})
    prediction_accuracy = np.mean(evaluations == y_vals_test)
    return prediction_accuracy