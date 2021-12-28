"""
Utility functions used in tools

Author: Venkanna Babu Guthula
Date: 03-07-2021
"""

import csv
import sys
from models import resunet_model, unet_model, segnet_model, unet_mini  # FCNs
import numpy as np
from keras import applications
import tensorflow as tf

# Load model
def select_model(args):
    if args.model == "unet":
        model = unet_model.unet(args)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "resunet":
        model = resunet_model.build_res_unet(args)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "unet_mini":
        model = unet_mini.UNet(args)
        if args.weights:
            model.load_weights(args.weights)
        return model

    # CNN Models (https://keras.io/api/applications)
    elif args.model == "xception":
        model = applications.xception.Xception(weights=None, input_shape=args.input_shape,
                                               classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "vgg16":
        model = applications.vgg16.VGG16(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "vgg19":
        model = applications.vgg19.VGG19(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "resnet50":
        model = applications.resnet50.ResNet50(weights=None, input_shape=args.input_shape,
                                               classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "resnet101":
        model = applications.ResNet101(weights=None, input_shape=args.input_shape,
                                       classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "resnet152":
        model = applications.ResNet152(weights=None, input_shape=args.input_shape,
                                       classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "resnet50v2":
        model = applications.ResNet50V2(weights=None, input_shape=args.input_shape,
                                        classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "resnet101v2":
        model = applications.ResNet101V2(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "resnet152v2":
        model = applications.ResNet152V2(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "inceptionv3":
        model = applications.InceptionV3(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "inceptionresnetv2":
        model = applications.InceptionResNetV2(weights=None, input_shape=args.input_shape,
                                               classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "mobilenet":
        model = applications.MobileNet(weights=None, input_shape=args.input_shape,
                                       classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "mobilenetv2":
        model = applications.MobileNetV2(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "densenet121":
        model = applications.DenseNet121(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "densenet169":
        model = applications.DenseNet169(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "densenet201":
        model = applications.DenseNet201(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "nasanetmobile":
        model = applications.NASNetMobile(weights=None, input_shape=args.input_shape,
                                         classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    elif args.model == "nasanetlarge":
        model = applications.nasnet.NASNetLarge(weights=None, input_shape=args.input_shape,
                                                classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)

    else:
        print(args.model + "Model does not exist, select model from"
                           " unet, unet_mini, resunet, segnet, xception, vgg16, vgg19, resnet50,"
                           "resnet101, resnet152, resnet50v2, resnet101v2, resnet152v2, inceptionv3,"
                           " inceptionresnetv2, mobilenet, mobilenetv2, densenet121, densenet169,"
                           "densenet201, nasanetmobile,  nasanetlarge")
        sys.exit()

    return model


# Get image and label paths from csv file
def file_paths(csv_file):
    image_paths = []
    label_paths = []
    with open(csv_file, 'r', newline='\n') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            image_paths.append(row[0])
            label_paths.append(row[1])
    return image_paths, label_paths


# Find the radiometric resolution of an image and get max number of an image
# To rescale the input data
def rescaling_value(value):
    return pow(2, value) - 1


def deep_lulc_data(array):
    # converting deep globe lulc labels to single band array
    # https://competitions.codalab.org/competitions/18468#participate-get_starting_kit
    array = np.where(array > 128, 1, 0)
    array1 = array[0]
    array2 = array[1] * 2
    array3 = array[2] * 4
    array = array1 + array2 + array3
    final_array = np.where(array == 7, 1, array)
    print(np.unique(final_array))
    return final_array
