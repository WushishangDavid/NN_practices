import time
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np

import mnist_inference as infer
import mnist_train as train

EVAL_INTERVAL_SECS = 60

def evaluate(dataset):
    with tf.Graph().as_default() as g:

        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        y_test = (np.arange(infer.OUTPUT_SIZE) == y_test).astype(int)

        x = tf.placeholder(tf.float32, [
            x_test.shape[0],
            infer.IMAGE_SIZE,
            infer.IMAGE_SIZE,
            infer.NUM_CHANNELS], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, infer.OUTPUT_SIZE], name='y_input')
        validate_feed = {x: x_test, y_: y_test}

        # No need to regularize during evaluation
        y = infer.inference(x, None, False)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # # Restore the shadow variables to evaluate
        # variables_average = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        # # Generate dict for restoring
        # variables_to_restore = variables_average.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)

        saver = tf.train.Saver()

        while(True):
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if(ckpt and ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), the validation accuracy is %s" %(global_step, accuracy_score))

                else:
                    print("No checkpoint file found.")
                    return

            time.sleep(EVAL_INTERVAL_SECS)


def main(argc=None):
    evaluate(cifar10)


if(__name__=='__main__'):
    tf.app.run()





