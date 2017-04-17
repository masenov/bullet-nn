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
def model(X, w_h, w_h2, w_h3, w_h4, w_h5, w_o, p_keep_input, p_keep_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h2 = tf.nn.dropout(h, p_keep_hidden)
        h3 = tf.nn.relu(tf.matmul(h2, w_h2))
    with tf.name_scope("layer3"):
        h4 = tf.nn.dropout(h3, p_keep_hidden)
        h5 = tf.nn.relu(tf.matmul(h4, w_h3))
    with tf.name_scope("layer4"):
        h6 = tf.nn.dropout(h5, p_keep_hidden)
        h7 = tf.nn.relu(tf.matmul(h6, w_h4))
    with tf.name_scope("layer5"):
        h8 = tf.nn.dropout(h7, p_keep_hidden)
        h9 = tf.nn.relu(tf.matmul(h8, w_h5))
    with tf.name_scope("layer6"):
        h10 = tf.nn.dropout(h9, p_keep_hidden)
        return tf.matmul(h10, w_o)

config=tf.ConfigProto()
config.gpu_options.allow_growth = True

with  tf.Session(config=config) as sess:

    data = seqData()
    train_data = data[:9000000,:]
    test_data = data[9000000:,:]


    trX = train_data[:,:16]
    trY = train_data[:,16:]
    teX = test_data[:,:16]
    teY = test_data[:,16:]

    print ("Loaded data")

    X = tf.placeholder(tf.float32, shape=[None, 16])
    Y = tf.placeholder(tf.float32, shape=[None, 6])

    #Step 3 - Initialize weights
    w_h = init_weights([16, 325], "w_h")
    w_h2 = init_weights([325, 325], "w_h2")
    w_h3 = init_weights([325, 325], "w_h3")
    w_h4 = init_weights([325, 325], "w_h4")
    w_h5 = init_weights([325, 325], "w_h5")
    w_o = init_weights([325, 6], "w_o")
    #Step 4 - Add histogram summaries for weights
    tf.summary.histogram("w_h_summ", w_h)
    tf.summary.histogram("w_h2_summ", w_h2)
    tf.summary.histogram("w_h3_summ", w_h3)
    tf.summary.histogram("w_h4_summ", w_h4)
    tf.summary.histogram("w_h5_summ", w_h5)
    tf.summary.histogram("w_o_summ", w_o)

    #Step 5 - Add dropout to input and hidden layers
    p_keep_input = tf.placeholder("float", name="p_keep_input")
    p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

    #Step 6 - Create Model
    py_x = model(X, w_h, w_h2, w_h3, w_h4, w_h5, w_o, p_keep_input, p_keep_hidden)
    #Step 7 Create cost function
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.square(py_x - Y))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        #train_op = tf.train.RMSPropOptimizer(0.1, 0.9).minimize(cost)
        #train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
        train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
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
    for i in range(1000):
        for start, end in zip(range(0, len(trX), 12800), range(12800, len(trX)+1, 12800)):
           sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        summary, train_acc = sess.run([merged, cost], feed_dict={X: trX[0:100000], Y: trY[0:100000],
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        summary, test_acc = sess.run([merged, cost], feed_dict={X: teX, Y: teY,
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        # writer.add_summary(summary, i)  # Write summary
        print ('%.12f, %.12f' % (train_acc, test_acc))                   # Report the accuracy
        np.save('nn_weights/1w_h_'+str(i),w_h.eval())
        np.save('nn_weights/1w_h2_'+str(i),w_h2.eval())
        np.save('nn_weights/1w_h3_'+str(i),w_h3.eval())
        np.save('nn_weights/1w_h4_'+str(i),w_h4.eval())
        np.save('nn_weights/1w_h5_'+str(i),w_h5.eval())
        np.save('nn_weights/1w_o_'+str(i),w_o.eval())

