import tensorflow as tf

INPUT_SIZE = 784
OUTPUT_SIZE = 10
LAYER_NODE = 500


def get_weight_variabble(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=.1))
    if(regularizer != None):
       tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(x, regularizer):
    with tf.variable_scope('layer1'):
        W = get_weight_variabble([INPUT_SIZE, LAYER_NODE], regularizer)
        b = tf.get_variable("biases", [LAYER_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(x,W)+b)

    with tf.variable_scope('layer2'):
        W = get_weight_variabble([LAYER_NODE, OUTPUT_SIZE], regularizer)
        b = tf.get_variable("biases", [OUTPUT_SIZE], initializer=tf.constant_initializer(0.0))
        # DON'T add activation func here!
        layer2 = tf.matmul(layer1,W)+b

    return layer2