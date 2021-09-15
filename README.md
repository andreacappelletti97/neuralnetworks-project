# Artificial Neural Networks and Deep Learning 2020

# Homework 1 - Image Classification

## Dataset

We loaded the dataset from a folder on Google Drive. In order to be able to use the
flow_from_dataframe​ function correctly, we created a new folder called ​tt​ in which we
copied the ​test​ folder.
We created a new pandas Dataframe to store the image name and their respective labels,
obtained from the ​train_gt.json​ file.
We then shuffle the dataframe in order to be able to split the training set between actual
training and validation.

### Found 3900 validated image filenames belonging to 3 classes.

### Found 1714 validated image filenames belonging to 3 classes.

### Found 450 images belonging to 1 classes.

## Training

The training of the model, after several tests (VGG19, ResNet50V2, EfficientNetB7), has
been done using Transfer Learning (VGG16).
We load the model and define the layer of the Convolutional Neural Network that we will
use to train the model. We then define the loss function and fit the model (​ **10 epochs** ​)
Epoch 1/
488/488 [==============================] - 173s 354ms/step - loss: 3.4952 - accuracy: 0.5831 -
val_loss: 1.0234 - val_accuracy: 0.
Epoch 2/
488/488 [==============================] - 173s 354ms/step - loss: 0.3402 - accuracy: 0.8669 -
val_loss: 1.2734 - val_accuracy: 0.
Epoch 3/
488/488 [==============================] - 172s 352ms/step - loss: 0.1567 - accuracy: 0.9487 -
val_loss: 1.7199 - val_accuracy: 0.
Epoch 4/
488/488 [==============================] - 171s 349ms/step - loss: 0.1474 - accuracy: 0.9649 -
val_loss: 1.6053 - val_accuracy: 0.
Epoch 5/
488/488 [==============================] - 172s 353ms/step - loss: 0.1469 - accuracy: 0.9662 -
val_loss: 2.3059 - val_accuracy: 0.
Epoch 6/
488/488 [==============================] - 172s 353ms/step - loss: 0.1321 - accuracy: 0.9651 -
val_loss: 2.4162 - val_accuracy: 0.


Maglione Sandro - 10532096 | Cappelletti Andrea - 10529039
Epoch 7/
488/488 [==============================] - 172s 353ms/step - loss: 0.2128 - accuracy: 0.9590 -
val_loss: 2.1282 - val_accuracy: 0.
Epoch 8/
488/488 [==============================] - 172s 352ms/step - loss: 0.1310 - accuracy: 0.9679 -
val_loss: 2.4050 - val_accuracy: 0.
Epoch 9/
488/488 [==============================] - 171s 351ms/step - loss: 0.2487 - accuracy: 0.9487 -
val_loss: 2.6791 - val_accuracy: 0.
Epoch 10/
488/488 [==============================] - 171s 351ms/step - loss: 0.1976 - accuracy: 0.9585 -
val_loss: 3.8082 - val_accuracy: 0.
<tensorflow.python.keras.callbacks.History at 0x7f6e9006d198>

## Test

Once the model completed the training phase, we compute the prediction on the test set.
We compute the most likely prediction of the model based on the results of the training
phase.
We then match the obtained label predictions with the original test image names. Finally, we
create a new pandas dataframe containing the result of our prediction and we export it in
.cvs.


# Homework 2 - Image Segmentation

## Dataset

We loaded the dataset from a folder on Google Drive. In order to import all the data we
divided the paths:
PATH_DATA = pathlib.Path(​"./drive/MyDrive/Development_Dataset"​)
PATH_WORKING = pathlib.Path(​"./drive/MyDrive"​)
PATH_TRAINING = PATH_DATA / ​"Training"
PATH_TEST = PATH_DATA / ​"Test_Dev"
We created a new pandas Dataframe to store the image name and their respective labels.
We then shuffle the dataframe in order to be able to split the training set between actual
training and validation.
Found 90 training images
Found 120 test images
Once done we obtained:
72 images for training
18 images for validation
120 images to test
Then we defined the ​MeanIoU ​function in order to keep an eye on the metrics of evaluation.

## Training

The training of the model, after several tests with different epochs, has resulted optimally
with 50 epochs.
We load the model and define the layer of the Convolutional Neural Network that we will
use to train the model. We then define the loss function and fit the model (​ **50 epochs** ​)


Maglione Sandro - 10532096 | Cappelletti Andrea - 10529039
Epoch 1/
18/18 [==============================] - ETA: 0s - loss: 0.8729 -
iou_score: 0.2581 - f1-score: 0.3205INFO:tensorflow:Assets written to:
drive/MyDrive/12-15_14-48/model/assets
18/18 [==============================] - 139s 8s/step - loss: 0.8729 -
iou_score: 0.2581 - f1-score: 0.3205 - val_loss: 0.8600 - val_iou_score:
0.2223 - val_f1-score: 0.
Epoch 2/
18/18 [==============================] - 117s 6s/step - loss: 0.8118 -
iou_score: 0.3768 - f1-score: 0.4737 - val_loss: 0.8672 - val_iou_score:
0.1853 - val_f1-score: 0.
Epoch 3/
18/18 [==============================] - ETA: 0s - loss: 0.7466 -
iou_score: 0.5288 - f1-score: 0.6406INFO:tensorflow:Assets written to:
drive/MyDrive/12-15_14-48/model/assets
18/18 [==============================] - 138s 8s/step - loss: 0.7466 -
iou_score: 0.5288 - f1-score: 0.6406 - val_loss: 0.8222 - val_iou_score:
0.3626 - val_f1-score: 0.
Epoch 4/
18/18 [==============================] - ETA: 0s - loss: 0.6782 -
iou_score: 0.6187 - f1-score: 0.7350INFO:tensorflow:Assets written to:
drive/MyDrive/12-15_14-48/model/assets
18/18 [==============================] - 137s 8s/step - loss: 0.6782 -
iou_score: 0.6187 - f1-score: 0.7350 - val_loss: 0.7832 - val_iou_score:
0.4998 - val_f1-score: 0.
...

## Test

Once the model completed the training phase, we compute the prediction on the test set.
We compute the most likely prediction of the model based on the results of the training
phase.
We then match the obtained label predictions with the original test image names. Finally, we
create a new pandas dataframe containing the result of our prediction and we export it in
.json format..




