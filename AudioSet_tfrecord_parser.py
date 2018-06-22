import tensorflow as tf
import glob

def _parse_function(example_proto):
    contexts, features = tf.parse_single_sequence_example(
            example_proto,
            context_features={"video_id": tf.FixedLenFeature([], tf.string),
                              "labels": tf.VarLenFeature(tf.int64)},
                              sequence_features={'audio_embedding' : tf.FixedLenSequenceFeature([10], dtype=tf.string)
                              })


    decoded_features = tf.reshape(
            tf.cast(tf.decode_raw(features['audio_embedding'], tf.uint8), tf.float32), [-1, 128])
    labels = (tf.cast(
            tf.sparse_to_dense(contexts["labels"].values, (527,), 1,
                               validate_indices=False),
                               tf.bool))


    return decoded_features, labels # and the labels?


filepath = 'your_feature_file_path'
# Get a list of files
filenames = glob.glob(filepath + '/*.tfrecord')
dataset = tf.data.TFRecordDataset(filenames)
dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(64)
iterator = dataset.make_one_shot_iterator()
#iterator = dataset.make_initializable_iterator()
#next_element = iterator.get_next()
#train_op = model_and_optimizer(images, labels)


sess = tf.Session()
for _ in range(2):
    while True:
        try:
            # How can x fit into a model now? Is it returning the features and labels?
            x = sess.run(iterator.get_next())
            print(x) # This prints out the byte code at least :)
        except tf.errors.OutOfRangeError:
            break