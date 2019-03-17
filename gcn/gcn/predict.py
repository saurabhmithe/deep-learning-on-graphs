import tensorflow as tf

# load the graph structure from the ".meta" file into the current graph
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("model.mdl.meta")

# load the values of variables
with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint('./'))


