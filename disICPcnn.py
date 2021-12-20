import tensorflow as tf
import numpy as np
import h5py
import time

def next_batch(train_data, train_target, batch_size):
    index = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(index);
    batch_data = [];
    batch_target = [];
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]]);
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

tf=tf.compat.v1
tf.compat.v1.disable_eager_execution()
"""
ps:169.254.104.35:8080
worker1:169.254.90.82:8080
worker2:169.254.117.117:8080
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '169.254.104.35:8080',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '169.254.90.82:8080,169.254.117.117:8080',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_integer(
        'task_id', 0, 'Task id of the replica running the training.')

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(
        cluster_spec,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id)
    print("!!!!")
    if FLAGS.job_name == 'ps':
        server.join()
    print("!!!!")

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster_spec)):
        # 导入训练集和测试集
        file = h5py.File('/home/pi/Documents/ICP.h5', 'r')
        x_train = file['train_data'][:]
        x_train = np.array(x_train)
        y_train = file['train_label'][:]
        y_train = np.array(y_train)
        x_test = file['test_data'][:]
        x_test = np.array(x_test)
        y_test = file['test_label'][:]
        y_test = np.array(y_test)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        learning_rate = 0.001
        training_iters = 600
        batch_size = 256
        display_step = 10

        n_input = 784
        n_classes = 7
        dropout = 0.75
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
            # 1024 inputs, 7 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
        keep_prob = tf.placeholder(dtype=tf.float32)  # dropout (keep probability)
        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        global_step = tf.Variable(0)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #tf.summary.scalar("accuarcy", accuracy)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_id == 0),
                             logdir="train_log",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=60)

    with sv.prepare_or_wait_for_session(server.target) as sess:
        #writer = tf.summary.FileWriter("log", sess.graph)
        time_begin = time.time()
        step=0;
        while step < training_iters :
            batch_x, batch_y = next_batch(x_train, y_train, batch_size)

            _,step=sess.run([optimizer,global_step], feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
            # summary,loss, acc = sess.run([summary_op,cost, accuracy],
            #                               feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            #writer.add_summary(summary, step)
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        time_end = time.time()
        train_time = time_end - time_begin
        print('Training elapsed time:%f s' % train_time)

        # Calculate accuracy for test images
        test_batch_x, test_batch_y = next_batch(x_test, y_test, 600)
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: test_batch_x,
                                            y: test_batch_y,
                                            keep_prob: 1.}))

    sv.stop()

if __name__ == "__main__":
    tf.app.run()


#python3 disICPcnn.py --job_name=ps --task_id=0
#python3 disICPcnn.py --job_name=worker --task_id=0
#python3 disICPcnn.py --job_name=worker --task_id=1