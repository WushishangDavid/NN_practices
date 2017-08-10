import os
import tensorflow as tf
from keras.datasets import cifar10
import mnist_inference as infer
import numpy as np

MODEL_SAVE_PATH = "./"
MODEL_NAME = 'model.ckpt'

REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
BATCH_SIZE = 100
LEARNING_RATE_DECAY = 0.99
TRAINING_STEP = 10000
TRAINING_SIZE = 50000
NUM_BATCH = TRAINING_SIZE/BATCH_SIZE

# Should be small!!!!!!
LEARNING_RATE_BASE = 0.0015

def train(dataset):
    # format of pic is tensor
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            infer.IMAGE_SIZE,
            infer.IMAGE_SIZE,
            infer.NUM_CHANNELS],
                    name='x_input')
    y_ = tf.placeholder(tf.float32, [None, infer.OUTPUT_SIZE], name='y_input')

    # Network Construction
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = infer.inference(x, regularizer, True)
    global_step = tf.Variable(0, trainable=False)

    # Loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # Learning rate setting
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               NUM_BATCH,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    # Training op
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step = global_step)

    # EMA op
    variables_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_average_op = variables_average.apply(tf.trainable_variables())

    # Overall op
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    # Transform y_train to one-hot
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    #print(y_train.shape)
    y_train = (np.arange(infer.OUTPUT_SIZE)==y_train).astype(int)
    #print(y_train.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEP):
            headix = int((i%NUM_BATCH)*BATCH_SIZE)
            tailix = int((i%NUM_BATCH+1)*BATCH_SIZE)
            print("%s  %s"%(headix, tailix))
            xs = x_train[headix:tailix,:,:,:]
            ys = y_train[headix:tailix,:]

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})

            if(i%10 == 0):
                print("After %d training step(s), loss on training batch is %g" %(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)


def main(argv=None):
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    train(cifar10)


if(__name__=='__main__'):
    tf.app.run()
    exit()
