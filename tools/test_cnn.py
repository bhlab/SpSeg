import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.xception import preprocess_input  # as preprocess_input_xception
from skimage.transform import resize
import numpy as np
from tensorflow.keras.preprocessing import image

cnn_model = "D:/venkanna/temp1/Label-Pixels/trained_models/xception300_05_10_21.hdf5"
img = "D:/DIgiKam_test/species_crops/TIGER/I__00255.JPG___crop00_mdv4.0.jpg"

model = load_model(cnn_model)
patch_size = 224

_image = image.load_img(img, target_size=(patch_size, patch_size))
_img = image.img_to_array(_image)
fig, ax = plt.subplots(figsize=(9, 9))
img = ax.imshow(_img)
plt.show()

# _img = np.array(image)
# _img = resize(_img, (224, 224))
# _img = image.img_to_array(_img)

_img = np.expand_dims(_img, axis=0)



_img = preprocess_input(_img)
pred = model.predict(_img)
print(pred[0])
print(len(pred))
# final_label = np.argmax(pred)