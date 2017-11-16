import tensorflow as tf

print (tf.__version__)

with tf.device("/GPU:0"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True
    # config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    sess = tf.Session(config=config)

    # matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.],[2.]])
    # product = tf.matmul(matrix1, matrix2)
    # result = sess.run(product)

    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    result = sess.run(c)
    print (result)