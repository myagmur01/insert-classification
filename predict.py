
import os
import cv2
import imutils
import pathlib
import numpy as np
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


testPath = "data/outliers/train"
img_size = (160,160)
savedModel = "models/resnet50_finetuned.h5"


import numpy as np
test_image = image.load_img('dataset/single_predictioncat_or_dog_1.jpg' , test_size = (64 , 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis = 0)
result = cnn.predic(test_image)
training_set.class_indices

# tf.keras.Model.load_weights()
#
# checkpoint_dir = os.path.dirname("models")
# latest = tf.train.latest_checkpoint(checkpoint_dir)
#
# # Use untrained model to evaluate
# model = create_model()
# loss, acc = model.evaluate(test_images, test_labels)
# print('Untrained model, accuracy: {:5.2f}%'.format(100*acc))
#
# # Use checkpoints
# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(test_images, test_labels)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
#
# # Use h5 model graph
# new_model = tf.keras.models.load_model('models/resnet50_finetuned.h5')
# new_model.summary()
#
# loss, acc = new_model.evaluate(test_images, test_labels)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


# ------------------------------------------------

testImgPaths = list(paths.list_images(testPath))

for inx, path in enumerate(testImgPaths):

    img = cv2.imread(path)
    orig = img.copy()

    img = cv2.resize(img, img_size).astype("float32") / 255.0
    #img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    model = load_model(savedModel)
    (prob1, prob2) = model.predict(img)[0]

    label =  "good" if prob1 > prob2 else "bad"
    pred = prob1 if prob1 > prob2 else prob2
    label = "{}: {:.2f}%".format(label, pred * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imwrite("results/pred.png", output)

    # reset the testing generator and then use our trained model to
    # make predictions on the data
    print("[INFO] evaluating network...")
    testGen.reset()
    predIdxs = model.predict(x=testGen, steps=(totalTest // BS) + 1)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testGen.classes, predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    # plot the training loss and accuracy
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])





#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")

