
# import the necessary packages
import os

DATA_DIR = "data"
# derive the training, validation, and testing directories
TRAIN_DIR = os.path.sep.join([DATA_DIR, "train"])
VAL_DIR = os.path.sep.join([DATA_DIR, "validation"])
TEST_DIR = os.path.sep.join([DATA_DIR, "test"])

finuted_model_ckpt_path = "models/finetuned-cp-{epoch:02d}.ckpt"
scratch_model_ckpt_path = "models/scratch0=-cp-{epoch:02d}.ckpt"
saveFinetunedModel = "models/resnet50_finetuned.h5"
saveScratchModel = "models/scratch_model.h5"

history_base = "results/simple_model_loss_accuracy_curve.png"
history_finetuned = "results/finetuned_model_loss_accuracy_curve.png"
history_scratch_model = "results/scratch_model_loss_accuracy_curve.png"

num_classes = 2
channels = 3
num_folds = 10
BATCH_SIZE = 8
IMG_SIZE = (224, 224)

learning_rate = 0.0001
initial_epochs = 5
fine_tune_epochs = 10

class_mode = "binary"
last_activation = "sigmoid"
focal_loss = False







# # ----------------------------------------------------------------------------
# #
# # ----------------------------------------------------------------------------
#
# # initialize the training data augmentation object
# self.trainAug = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=30,
#                 zoom_range=0.05, # 0.15
#                 width_shift_range=0.1, #0.2
#                 height_shift_range=0.1, #0.2
#                 shear_range=0.05, #0.15
#                 horizontal_flip=True,
#                 vertical_flip=True,
#                 fill_mode="nearest"
#                 )
#
# # initialize the validation (and testing) data augmentation object
# self.valAug = tf.keras.preprocessing.image.ImageDataGenerator()
#
# #mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# #trainAug.mean = mean
# #valAug.mean = mean
#
# # initialize the training generator
# self.trainGen = self.trainAug.flow_from_directory(
#                 self.TRAIN_DIR,
#                 class_mode=self.class_mode,
#                 target_size=self.IMG_SIZE,
#                 color_mode="rgb",
#                 shuffle=True,
#                 batch_size=self.BATCH_SIZE
#                 )
# # initialize the validation generator
# self.valGen = self.valAug.flow_from_directory(
#                 self.VAL_DIR,
#                 class_mode=self.class_mode,
#                 target_size=self.IMG_SIZE,
#                 color_mode="rgb",
#                 shuffle=True,
#                 batch_size=self.BATCH_SIZE
#                 )

