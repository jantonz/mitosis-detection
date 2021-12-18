from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns


#######################################
#### Train net with VGG16 + denses ####
#######################################

src_path_train = "../dataset/train/"
src_path_test = "../dataset/train/"

image_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.20,
)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)


batch_size = 64
train_generator = image_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(70, 70),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)
valid_generator = image_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(70, 70),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=42,
)
test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(70, 70),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42,
)

base_model = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(70, 70, 3),
)
base_model.trainable = False
base_model.summary()

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile()
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["AUC", "accuracy"]
)
model.summary()

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(70, 70, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(16, activation="relu"))
# model.add(Dense(2, activation="softmax"))
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC", "accuracy"])

history = model.fit(
    train_generator,
    validation_data=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    epochs=100,
)

# create a HDF5 file 'my_model.h5'
model.save("model.h5")


#######################################
#### Plot training and other plots ####
#######################################

# plot model summary
plot_model(model)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# summarize history for AUC
plt.plot(history.history["auc"])
plt.plot(history.history["val_auc"])
plt.title("model auc")
plt.ylabel("auc")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

y_pred = model.predict(test_generator)
y = np.argmax(y_pred, axis=1)
print("Confusion Matrix")
print(confusion_matrix(test_generator.classes, y))

ConfusionMatrixDisplay.from_predictions(test_generator.classes, y)
plt.show()

print("Classification Report")
target_names = ["not mitosis", "mitosis"]
clf_report = classification_report(
    test_generator.classes, y, target_names=target_names, output_dict=True
)

sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)


#######################################
#### Visualize images within VGG16 ####
#######################################

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims


# load the model
viz_model = VGG16()
# redefine viz_model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17]
outputs = [viz_model.layers[i].output for i in ixs]
viz_model = Model(inputs=viz_model.inputs, outputs=outputs)
# load the image with the required shape
# convert the image to an array
img = load_img(f"../dataset/train/1/A03_00Aa_1_true.png", target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = viz_model.predict(img)
# plot the output from each block
square = 4
for j, fmap in enumerate(feature_maps):
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        plt.figure(figsize=(64, 64))
        for _ in range(square):

            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])

            # plot filter channel in grayscale
            plt.imshow(fmap[0, :, :, ix - 1], cmap="viridis")
            ix += 1
    # show the figure
    plt.show()
