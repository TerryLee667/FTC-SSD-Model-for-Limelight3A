import tensorflow as tf

tfrecord_path = '/home/user/data/train.tfrecord'

for raw_record in tf.data.TFRecordDataset([tfrecord_path]).take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    keys = list(example.features.feature.keys())
    print("TFRecord 中的键名:", keys)