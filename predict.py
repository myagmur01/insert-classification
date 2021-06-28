
import config
import numpy as np
import tensorflow as tf
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Total number of test images
totalTest = len(list(paths.list_images(config.TEST_DIR)))

# Call trained model
model = tf.keras.models.load_model(config.saveFinetunedModel)

testAug = tf.keras.preprocessing.image.ImageDataGenerator()
testGen = testAug.flow_from_directory(
            config.TEST_DIR,
            class_mode=config.class_mode,
            target_size=config.IMG_SIZE,
            color_mode="rgb",
            shuffle=False,
            batch_size=config.BATCH_SIZE
            )

print("[INFO] evaluating network...")
testGen.reset()

predIdxs = model.predict(x=testGen, steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs,target_names=testGen.class_indices.keys()))

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

