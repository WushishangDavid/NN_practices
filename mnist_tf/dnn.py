import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 784
LABEL_SIZE = 10

LAYER_NODE = 500
BATCH_SIZE = 100
MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEP = 30000

def inference(x, avg_class, W1, b1, W2, b2):
    if(avg_class==None):
        layer1 = tf.nn.relu(tf.matmul(x,W1)+b1)
        return tf.matmul(layer1,W2)+b2
    # Case when Moving Average is used
    else:
        layer1 = tf.nn.relu(tf.matmul(x, avg_class.average(W1)) + avg_class.average(b1))
        return tf.matmul(layer1, avg_class.average(W2)) + avg_class.average(b2)
        

def train(mnist):
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])
    # First layer
    W1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE, LAYER_NODE], stddev=.1))
    b1 = tf.Variable(tf.constant(0.1, shape = [LAYER_NODE]))
    # Output layer
    W2 = tf.Variable(tf.truncated_normal([LAYER_NODE, LABEL_SIZE], stddev=.1))
    b2 = tf.Variable(tf.constant(0.1, shape = [LABEL_SIZE]))
    
    # y for training
    y = inference(x, None, W1, b1, W2, b2)
    
    # Use moving average
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # Calculate and keep the MA values for every trainable var (not actually used in training)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # Use MA values in evaluation
    average_y = inference(x, variable_averages, W1, b1, W2, b2)
    
    # 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(W1) + regularizer(W2)
    loss = cross_entropy_mean + regularization
    
    # Learning rate setting
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    correct_prediction = tf.cast(tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_:mnist.test.labels}
        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d steps, validation accuracy used average model is: %g" % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_:ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d steps, testing accuracy used average model is: %g" % (TRAINING_STEP, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)
    
if(__name__ == '__main__'):
    tf.app.run()
    exit()