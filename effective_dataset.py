import tensorflow as tf

def input_fn(batch_size):
    files = tf.data.Dataset.list_files(FLAGS.data_dir)
    dataset = tf.data.TFRecordDataset(files.num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = data.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
    
def input_fn_optimized(batch_size):
    files = tf.data.Dataset.list_files(FLAGS.data_dir)
    
    def tfrecord_dataset(filename):
        buffer_size = 8*1024*1024 #8M per file
        return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
        
    dataset = files.apply(tf.contrib.data.parallel_interleave(tfrecord_dataset, cycle_length=32, sloppy=True))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, NUM_EPOCHS))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parser_fn, batch_size, num_parallel_batchs=tf.data.experimental.AUTOTUNE))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
    
