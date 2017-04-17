import tensorflow as tf
import numpy as np
from api import *
from time import gmtime, strftime

# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)

# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

# This network is the same as the previous one except with an extra hidden layer + dropout
def model(X, w_h, w_h2, w_h3, w_o, p_keep_input, p_keep_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope("layer3"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h3))
    with tf.name_scope("layer4"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o)

config=tf.ConfigProto()
config.gpu_options.allow_growth = True

with  tf.Session(config=config) as sess:

    data = seqData()
    train_data = data[0:90000,:]
    test_data = data[90000:,:]


    trX = train_data[:,:64]
    trY = train_data[:,64:]
    teX = test_data[:,:64]
    teY = test_data[:,64:]

    print ("Loaded data")

    X = tf.placeholder(tf.float32, shape=[None, 64])
    Y = tf.placeholder(tf.float32, shape=[None, 6])

    #Step 3 - Initialize weights
    w_h = init_weights([64, 185], "w_h")
    w_h2 = init_weights([185, 185], "w_h2")
    w_h3 = init_weights([185, 185], "w_h3")
    w_o = init_weights([185, 6], "w_o")

    #Step 4 - Add histogram summaries for weights
    tf.summary.histogram("w_h_summ", w_h)
    tf.summary.histogram("w_h2_summ", w_h2)
    tf.summary.histogram("w_h3_summ", w_h3)
    tf.summary.histogram("w_o_summ", w_o)

    #Step 5 - Add dropout to input and hidden layers
    p_keep_input = tf.placeholder("float", name="p_keep_input")
    p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

    #Step 6 - Create Model
    py_x = model(X, w_h, w_h2, w_h3, w_o, p_keep_input, p_keep_hidden)

    #Step 7 Create cost function
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.square(py_x - Y))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        #train_op = tf.train.RMSPropOptimizer(0.1, 0.9).minimize(cost)
        #train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
        train_op = tf.train.AdamOptimizer().minimize(cost)
        # Add scalar summary for cost tensor
        tf.summary.scalar("cost", cost)


    #Step 8 Measure accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(Y, py_x) # Count correct predictions
        acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
        # Add scalar summary for accuracy tensor
        tf.summary.scalar("accuracy", acc_op)


    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #Step 9 Create a session
        # Step 10 create a log writer. run 'tensorboard --logdir=./logs/nn_logs
    # writer = tf.summary.FileWriter("nn_logs/" + strftime("%Y-%m-%d %H:%M:%S", gmtime()), sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    # Step 11 you need to initialize all variables
    tf.global_variables_initializer().run()

    #Step 12 train the  model
    for i in range(2000):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 1.0, p_keep_hidden: 0.5})
        summary, train_acc = sess.run([merged, cost], feed_dict={X: trX, Y: trY,
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        summary, test_acc = sess.run([merged, cost], feed_dict={X: teX, Y: teY,
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        # writer.add_summary(summary, i)  # Write summary
        print ('%.12f, %.12f' % (train_acc, test_acc))                   # Report the accuracy
    np.save('1w_h',w_h.eval())
    np.save('1w_h2',w_h2.eval())
    np.save('1w_h3',w_h3.eval())
    np.save('1w_o',w_o.eval())


