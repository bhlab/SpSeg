# load yolov3 model and perform object detection
import argparse
import glob
import os
import statistics
import sys
import time
import warnings

import humanfriendly
import numpy as np
from tqdm import tqdm

from models.ct_utils import truncate_float
import visualization.visualization_utils as viz_utils

import tensorflow as tf
from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.xception import preprocess_input  # as preprocess_input_xception
from skimage.transform import resize
import shutil


print('TensorFlow version:', tf.__version__)
print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())


# %% Classes
class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings


class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    the MegaDetector (TF). The inference batch size is set to 1; code needs to be modified
    to support larger batch sizes, including resizing appropriately.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # An enumeration of failure reasons
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '3': 'vehicle'  # available in megadetector v4+
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads model from model_path and starts a tf.Session with this graph. Obtains
        input and output tensor handles."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.compat.v1.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        #Not added for additional boxes
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(tf_coords):
        """Converts coordinates from the model's output format [y1, x1, y2, x2] to the
        format used by our API and MegaDB: [x1, y1, width, height]. All coordinates
        (including model outputs) are normalized in the range [0, 1].
        Args:
            tf_coords: np.array of predicted bounding box coordinates from the TF detector,
                has format [y1, x1, y2, x2]
        Returns: list of Python float, predicted bounding box coordinates [x1, y1, width, height]
        """
        # change from [y1, x1, y2, x2] to [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """From [x1, y1, width, height] to [y1, x1, y2, x2], where x1 is x_min, x2 is x_max
        This is an extraneous step as the model outputs [y1, x1, y2, x2] but were converted to the API
        output format - only to keep the interface of the sync API.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.
        Args:
            model_path: .pb file of the model.
        Returns: the loaded graph.
        """
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id,
                                      detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.
        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal
        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result


# %% Main function


def load_and_run_detector(model_file, image_file_names, output_dir, cnn_model,
                          render_confidence_threshold=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
                          crop_images=False, sort_data=False, grid_dir=2):
    """Load and run detector on target images, and visualize the results."""
    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    start_time = time.time()
    tf_detector = TFDetector(model_file)
    tf_classifier = load_model(cnn_model)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    detection_results = []
    time_load = []
    time_infer = []

    # Dictionary mapping output file names to a collision-avoidance count.
    #
    # Since we'll be writing a bunch of files to the same folder, we rename
    # as necessary to avoid collisions.
    output_filename_collision_counts = {}

    def input_file_to_detection_file(fn, crop_index=-1):
        """Creates unique file names for output files.
        This function does 3 things:
        1) If the --crop flag is used, then each input image may produce several output
            crops. For example, if foo.jpg has 3 detections, then this function should
            get called 3 times, with crop_index taking on 0, 1, then 2. Each time, this
            function appends crop_index to the filename, resulting in
                foo_crop00_detections.jpg
                foo_crop01_detections.jpg
                foo_crop02_detections.jpg
        2) If the --recursive flag is used, then the same file (base)name may appear
            multiple times. However, we output into a single flat folder. To avoid
            filename collisions, we prepend an integer prefix to duplicate filenames:
                foo_crop00_detections.jpg
                0000_foo_crop00_detections.jpg
                0001_foo_crop00_detections.jpg
        3) Prepends the output directory:
                out_dir/foo_crop00_detections.jpg
        Args:
            fn: str, filename
            crop_index: int, crop number
        Returns: output file path
        """
        fn = os.path.basename(fn).lower()
        name, ext = os.path.splitext(fn)
        if crop_index >= 0:
            name += '_crop{:0>2d}'.format(crop_index)
        fn = '{}{}{}'.format(name, ImagePathUtils.DETECTION_FILENAME_INSERT, '.jpg')
        if fn in output_filename_collision_counts:
            n_collisions = output_filename_collision_counts[fn]
            fn = '{:0>4d}'.format(n_collisions) + '_' + fn
            output_filename_collision_counts[fn] += 1
        else:
            output_filename_collision_counts[fn] = 0
        fn = os.path.join(output_dir, fn)
        return fn

    def cnn_predictions(images_cropped, species_prob):
        predictions = []
        for _image in images_cropped:
            _img = resize(_image, (224, 224))
            # _img = tf.image.resize(_image, (224, 224))
            _img = np.expand_dims(_img, axis=0)
            _img = preprocess_input(_img)
            pred = tf_classifier.predict(_img)
            final_label = np.argmax(pred)
            if pred[0][final_label] > species_prob:
                predictions.append(final_label)

        return predictions

    def create_dir_l1(out_dir, folder, file_name):
        path = os.path.join(out_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_name)

        return path

    def create_dir_l2(out_dir, species_index, file_name):
        # path_l1 = os.path.join(out_dir, "animal")
        # if not os.path.exists(path_l1):
        #     os.mkdir(path_l1)
        species_list = ["barking _deer",  "birds", "buffalo", "spotted_deer",
                        "four_horned_antelope", "common_palm_civet", "cow",
                        "dog", "gaur", "goat", "golden_jackal", "hare",
                        "hyena", "indian_fox", "indian_pangolin", "indian_porcupine",
                        "jungle_cat", "jungle_fowls", "langur", "leopard", "macaque",
                        "nilgai", "palm_squirrel", "indian_peafowl", "ratel",
                        "rodents", "mongooses", "rusty_spotted_cat", "sambar", "sheep",
                        "sloth_bear", "small_indian_civet", "tiger", "wild_boar",
                        "wild_dog", "indian_wolf"]

        path_l2 = os.path.join(out_dir, species_list[species_index])
        if not os.path.exists(path_l2):
            os.mkdir(path_l2)
        path_l2 = os.path.join(path_l2, file_name)

        return path_l2

    for im_file in tqdm(image_file_names):
        # Create output directory and file name
        im_path = os.path.normpath(im_file)
        im_path_list = im_path.split(os.sep)
        file_name = im_path_list[-1]
        if grid_dir == 1:
            out_file_dir = os.path.join(output_dir, im_path_list[-2])
        elif grid_dir == 2:
            out_file_dir = os.path.join(output_dir, im_path_list[-3], im_path_list[-2])
        elif grid_dir == 3:
            out_file_dir = os.path.join(output_dir, im_path_list[-4], im_path_list[-3], im_path_list[-2])
        else:
            out_file_dir = output_dir
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
        #print(out_file_dir, file_name)

        try:
            start_time = time.time()

            image = viz_utils.load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            result = tf_detector.generate_detections_one_image(image, im_file)
            detection_results.append(result)
            elapsed = time.time() - start_time
            time_infer.append(elapsed)

        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(im_file, e))
            continue


        # Sort data
        try:
            # print(result)
            if sort_data:

                if len(result['detections']) == 0:
                    copy_dir = create_dir_l1(out_file_dir, 'blank', file_name)
                    shutil.copyfile(im_file, copy_dir)

                else:
                    classes = []
                    for detect in result['detections']:
                        if detect['conf'] >= render_confidence_threshold:  # sorting threshold
                            classes.append(detect['category'])
                    if len(classes) == 0:
                        copy_dir = create_dir_l1(out_file_dir, 'blank', file_name)
                        shutil.copyfile(im_file, copy_dir)
                    elif len(classes) >= 1:
                        if "3" in classes and "2" not in classes and "1" not in classes:  # Vehicle
                            copy_dir = create_dir_l1(out_file_dir, 'vehicle', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        elif "2" in classes and "3" not in classes and "1" not in classes:  # Human
                            copy_dir = create_dir_l1(out_file_dir, 'human', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        elif "2" in classes and "3" in classes and "1" not in classes:  # Human and Vehicle
                            copy_dir = create_dir_l1(out_file_dir, 'human_vehicle', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        elif "1" in classes and "2" not in classes and "3" not in classes:  # Animal
                            images_cropped = viz_utils.square_crop_image(result['detections'], im_file)
                            detections = cnn_predictions(images_cropped, species_prob=0.5)
                            detections = np.unique(np.array(detections))
                            # print("unique classes: ", detections)
                            if len(detections) == 0:
                                copy_dir = create_dir_l2(out_file_dir, -1, file_name)
                                shutil.copyfile(im_file, copy_dir)
                            else:
                                for cls in detections:
                                    copy_dir = create_dir_l2(out_file_dir, cls, file_name)
                                    shutil.copyfile(im_file, copy_dir)
                        elif "1" in classes and "2" in classes and "3" not in classes:  # Animal and Human
                            copy_dir = create_dir_l1(out_file_dir, 'animal_human', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        elif "1" in classes and "3" in classes and "2" not in classes:  # Animal and Vehicle
                            copy_dir = create_dir_l1(out_file_dir, 'animal_vehicle', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        elif "1" in classes and "2" in classes and "3" in classes:  # Animal, Vehicle and Human
                            copy_dir = create_dir_l1(out_file_dir, 'animal_human_vehicle', file_name)
                            shutil.copyfile(im_file, copy_dir)
        except Exception as e:
            print('Sorting data failed', e)
            continue

        """
                try:
            if crop_images:

                images_cropped = viz_utils.crop_image(result['detections'], image)

                for i_crop, cropped_image in enumerate(images_cropped):
                    output_full_path = input_file_to_detection_file(im_file, i_crop)
                    cropped_image.save(output_full_path)

            else:

                # image is modified in place
                viz_utils.render_detection_bounding_boxes(result['detections'], image,
                                                          label_map=TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
                                                          confidence_threshold=render_confidence_threshold)
                output_full_path = input_file_to_detection_file(im_file)
                image.save(output_full_path)

        except Exception as e:
            print('Visualizing results on the image {} failed. Exception: {}'.format(im_file, e))
            continue
        """


    # ...for each image

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available'
        std_dev_time_infer = 'not available'
    print('On average, for each image,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))


# %% Command-line driver


def main():
    parser = argparse.ArgumentParser(
        description='Module to run a TF animal detection model on images')
    parser.add_argument(
        'detector_file',
        help='Path to .pb TensorFlow detector model file')
    group = parser.add_mutually_exclusive_group(required=True)  # must specify either an image file or a directory
    group.add_argument(
        '--image_file',
        help='Single file to process, mutually exclusive with --image_dir')
    group.add_argument(
        '--image_dir',
        help='Directory to search for images, with optional recursion by adding --recursive')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir')
    parser.add_argument(
        '--output_dir',
        help='Directory for output images (defaults to same as input)')
    parser.add_argument(
        '--threshold',
        type=float,
        default=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
        help=('Confidence threshold between 0 and 1.0; only render boxes above this confidence'
              ' (but only boxes above 0.1 confidence will be considered at all)'))
    parser.add_argument(
        '--crop',
        default=False,
        action="store_true",
        help=('If set, produces separate output images for each crop, '
              'rather than adding bounding boxes to the original image'))
    parser.add_argument(
        '--sort_data',
        default=False,
        action="store_true",
        help='Sort data: Create folder for each class and sort automatically')
    parser.add_argument(
        '--grid_dir',
        type= int,
        default=2,
        help='Grid directory: Specify the levels of folders from inDir to the image from 1 to 3')
    parser.add_argument(
        '--cnn_model',
        help='Path to .pb TensorFlow detector model file')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'detector_file specified does not exist'
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        image_file_names = ImagePathUtils.find_images(args.image_dir, args.recursive)

    print('Running detector on {} images...'.format(len(image_file_names)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            # but for a single image, args.image_dir is also None
            args.output_dir = os.path.dirname(args.image_file)

    load_and_run_detector(model_file=args.detector_file,
                          image_file_names=image_file_names,
                          output_dir=args.output_dir,
                          cnn_model=args.cnn_model,
                          render_confidence_threshold=args.threshold,
                          crop_images=args.crop, sort_data=args.sort_data,
                          grid_dir=args.grid_dir)


if __name__ == '__main__':
    main()
