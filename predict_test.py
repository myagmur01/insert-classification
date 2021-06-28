
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical



# test_image = image.load_img('dataset/single_predictioncat_or_dog_1.jpg' ,
#                             test_size = (64 , 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image , axis = 0)
# result = cnn.predic(test_image)
# training_set.class_indices

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

# testImgPaths = list(paths.list_images(testPath))
#
# for inx, path in enumerate(testImgPaths):
#
#     img = cv2.imread(path)
#     orig = img.copy()
#
#     img = cv2.resize(img, img_size).astype("float32") / 255.0
#     #img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#
#     model = load_model(savedModel)
#     (prob1, prob2) = model.predict(img)[0]
#
#     label =  "good" if prob1 > prob2 else "bad"
#     pred = prob1 if prob1 > prob2 else prob2
#     label = "{}: {:.2f}%".format(label, pred * 100)
#
#     # draw the label on the image
#     output = imutils.resize(orig, width=400)
#     cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 255, 0), 2)
#
#     # show the output image
#     cv2.imwrite("results/pred.png", output)



# # plot the training loss and accuracy
# N = config.fine_tune_epochs
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])

#Retrieve a batch of images from the test set
# image_batch, label_batch = test_dataset.as_numpy_iterator().next()
# predictions = model.predict_on_batch(image_batch).flatten()
#
# # Apply a sigmoid since our model returns logits
# predictions = tf.nn.sigmoid(predictions)
# predictions = tf.where(predictions < 0.5, 0, 1)
#
# print('Predictions:\n', predictions.numpy())
# print('Labels:\n', label_batch)
#
# plt.figure(figsize=(10, 10))
# for i in range(9):
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(image_batch[i].astype("uint8"))
#   plt.title(class_names[predictions[i]])
#   plt.axis("off")