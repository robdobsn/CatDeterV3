import re
import tensorflow as tf
import os
import cv2
import numpy as np
from Utils import DebugTimer

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,model_dir,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

class ImageRecogniser():
    def __init__(self, model_dir):
        # Creates graph from saved GraphDef.
        self.model_dir = model_dir
        self.debugTimer = DebugTimer(["ConvertImage", "RecogniseImage","Lookup"])

    def start(self):
        return self.create_graph()

    def create_graph(self):
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(os.path.join(
                self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            return True
        return False

    def recogniseImage(self, sess, image, num_top_predictions):

        self.debugTimer.start(0)
        # Convert image
        img2 = cv2.resize(image, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        # Numpy array
        np_image_data = np.asarray(img2)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        # maybe insert float convertion here - see edit remark!
        np_final = np.expand_dims(np_image_data, axis=0)
        self.debugTimer.end(0)

        self.debugTimer.start(1)
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,{'Mul:0': np_final})
        predictions = np.squeeze(predictions)
        self.debugTimer.end(1)

        self.debugTimer.start(2)
        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup(self.model_dir)

        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        node_id = top_k[0]
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        self.debugTimer.end(2)
        if score > 0.2 and "iamese" in human_string:
            return ("bad", 100)
        return ("good", 100)
