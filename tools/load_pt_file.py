import tensorflow as tf
print(tf.__version__)
model_path = './models/md_v4_1_0.pb'


def __load_model(model_path):
    """Loads a detection model (i.e., create a graph) from a .pb file.

    Args:
        model_path: .pb file of the model.

    Returns: the loaded graph.
    """
    print('TFDetector: Loading graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('TFDetector: Detection graph loaded.')

    return detection_graph


model = __load_model(model_path)

print("model loaded")