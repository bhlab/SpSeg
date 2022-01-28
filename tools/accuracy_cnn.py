import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from models import lp_utils as lu
import argparse


def cnn_predict(args):
    test_image_paths, test_label = lu.file_paths(args.csv_paths)
    model = load_model(args.weights)
    net = args.model

    # Importing preprocess function for required models
    if net == "xception":
        from tensorflow.keras.applications.xception import preprocess_input
    elif net == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif net == "vgg19":
        from tensorflow.keras.applications.vgg19 import preprocess_input
    elif net == "resnet50" or net == "resnet101" or net == "resnet152":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif net == "resnet50v2" or net == "resnet101v2" or net == "resnet152v2":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif net == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif net == "inceptionresnetv2":
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    elif net == "mobilenet" or net == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet import preprocess_input
    elif net == "densenet121" or net == "densenet169" or net == "densenet201":
        from tensorflow.keras.applications.densenet import preprocess_input
    elif net == "nasanetmobile" or net == "nasanetlarge":
        from tensorflow.keras.applications.nasnet import preprocess_input
    else:
        print(net + " model does not exist, select model from xception, vgg16, vgg19, resnet50,"
              " resnet101, resnet152, resnet50v2, resnet101v2, resnet152v2, inceptionv3,"
              " inceptionresnetv2, densenet201, nasanetmobile,  nasanetlarge")

    predictions = []
    labels = []
    for i, img in enumerate(test_image_paths):
        try:
            _img = image.load_img(img, target_size=(224, 224))
            _img = image.img_to_array(_img)
            _img = np.expand_dims(_img, axis=0)
            _img = preprocess_input(_img)
            pred = model.predict(_img)
            final_label = np.argmax(pred)
            predictions.append(final_label)
            labels.append(test_label[i])
            print(i, final_label, test_label[i])
        except:
            print(img, " Not found")

    y = np.array(labels).astype(int)
    y_pred = np.array(predictions)

    print("List of classes: ", np.unique(y))
    print("List of predictions: ", np.unique(y_pred))

    acc = accuracy_score(y, y_pred)
    print("Overal Accuracy: ", round(acc, 3), "\n")
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm, "\n")
    cm_multi = multilabel_confusion_matrix(y, y_pred)
    for j in range(len(cm_multi)):
        print("Class: " + str(j))
        iou = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][0][1] + cm_multi[j][1][0])
        f1 = (2 * cm_multi[j][1][1]) / (2 * cm_multi[j][1][1] + cm_multi[j][0][1] + cm_multi[j][1][0])
        precision = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][0][1])
        recall = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][1][0])
        print("IoU Score: ", round(iou, 3))
        print("F1-Measure: ", round(f1, 3))
        print("Precision: ", round(precision, 3))
        print("Recall: ", round(recall, 3), "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model (from unet, resunet, or segnet)")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    args = parser.parse_args()
    cnn_predict(args)
