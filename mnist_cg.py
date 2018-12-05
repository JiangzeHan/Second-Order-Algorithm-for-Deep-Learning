import numpy as np
import tensorflow as tf
from tqdm import tqdm

from optimizers import cg

# CG for a toy model 
# MNIST Dataset of Handwritten Digits

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float64').reshape(x_train.shape[0], 28, 28, 1) / 255 - 0.5
x_test = x_test.astype('float64').reshape(x_test.shape[0], 28, 28, 1) / 255 - 0.5

n_training = x_train.shape[0]
n_epochs = 20
n_batches = 100
iters_per_epoch = int(n_training / n_batches)
learning_rate = 0.01

features = tf.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1), name='features')
labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='labels')

conv1 = tf.layers.conv2d(inputs=features, filters=4, kernel_size=2, padding='same', activation=tf.nn.softplus, name='conv1')
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, name='pool1')
conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=2, padding='same', activation=tf.nn.softplus, name='conv2')
pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=2, strides=2, name='pool2')
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 16])
dense1 = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.softplus, name='dense1')
dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.softplus, name='dense2')
logits = tf.layers.dense(inputs=dense2, units=10, name='logits')

classes = tf.argmax(input=logits, axis=1)
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, tf.cast(classes, tf.int32)), tf.float32))

train_loss_summary = tf.summary.scalar('loss', loss)
train_acc_summary = tf.summary.scalar('acc', accuracy)
test_loss_summary = tf.summary.scalar('val_loss', loss)
test_acc_summary = tf.summary.scalar('val_acc', accuracy)

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
grads = tf.gradients(loss, params)

grads_ = [tf.Variable(np.zeros(g.get_shape()), trainable=False, name='gradients_') for g in grads]
deltas_ = [tf.Variable(np.zeros(g.get_shape()), trainable=False, name='deltas_') for g in grads]

with tf.name_scope('train'):
    update_directions = [cg(g, global_step_tensor, g_, d_) for g, g_, d_ in zip(grads, grads_, deltas_)]
    grad_updates = [g_.assign(g) for g, g_ in zip(grads, grads_)]
    delta_updates = [d_.assign(d) for d, d_ in zip(update_directions, deltas_)]
    weight_updates = [p.assign_add(learning_rate * up) for p, up in zip(params, update_directions)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    writer = tf.summary.FileWriter('logs_/cg', sess.graph)

    for i in range(n_epochs):
        for j in tqdm(range(iters_per_epoch), desc='epochs %s' % i):
            sample_list = np.random.choice(n_training, n_batches)
            x_batch = x_train[sample_list]
            y_batch = y_train[sample_list]

            feed_dict = {
                features: x_batch,
                labels: y_batch
            }

            sess.run([grad_updates, delta_updates, weight_updates], feed_dict=feed_dict)

        feed_dict = {
            features: x_train,
            labels: y_train
        }

        train_loss, train_acc = sess.run([train_loss_summary, train_acc_summary], feed_dict=feed_dict)
        writer.add_summary(train_loss, i)
        writer.add_summary(train_acc, i)
        writer.flush()

        feed_dict = {
            features: x_test,
            labels: y_test
        }

        test_loss, test_acc = sess.run([test_loss_summary, test_acc_summary], feed_dict=feed_dict)
        writer.add_summary(test_loss, i)
        writer.add_summary(test_acc, i)
        writer.flush()