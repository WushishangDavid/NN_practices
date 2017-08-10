import tensorflow as tf

# Input format
INPUT_SIZE = 32*32
OUTPUT_SIZE = 10

IMAGE_SIZE = 32
NUM_CHANNELS = 3

# Parameters for CNN
CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 32
CONV2_SIZE = 5

CONV3_DEEP = 64
CONV3_SIZE = 5

FC_SIZE = 64

def inference(x, regularizer, train):
    with tf.variable_scope('layer1_conv1'):
        conv1_W = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                  initializer=tf.truncated_normal_initializer(stddev=0.0001))
        conv1_b = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))

    with tf.variable_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer3_conv2'):
        conv2_W = tf.get_variable("weight", shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv2_b = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_W, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

    with tf.variable_scope('layer4_pool2'):
        pool2 = tf.nn.avg_pool(relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer5_conv3'):
        conv3_W = tf.get_variable("weight", shape=[CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv3_b = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_W, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))

    with tf.variable_scope('layer6_pool3'):
        pool3 = tf.nn.avg_pool(relu3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    pool_shape = pool3.get_shape().as_list()
    # Shape[0] = BATCH_SIZE
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

    with tf.variable_scope('layer7_fc1'):
        fc1_W = tf.get_variable("weight", shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if(regularizer!=None):
            tf.add_to_collection('losses', regularizer(fc1_W))
        fc1_b = tf.get_variable("bias", shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_W)+fc1_b)
        # Dropout is only used in FC
        if(train):
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer8_fc2'):
        fc2_W = tf.get_variable("weight", shape=[FC_SIZE, OUTPUT_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if (regularizer != None):
            tf.add_to_collection('losses', regularizer(fc2_W))
        fc2_b = tf.get_variable("bias", shape=[OUTPUT_SIZE], initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1,fc2_W)+fc2_b

    return logit



