

import config
import argparse
import numpy as np

import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adagrad

from insertnet import InsertNet
from utils.focal_loss import BinaryFocalLoss
from utils.get_class_weihgts import calculate_class_weights
from utils.plot_training import plot_accuracy_loss_curve

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--approach", required=True,
                      help="if training from scratch pass 'scratch' else 'finetune'")
# ap.add_argument("-m", "--model", required=True, help="path to output model")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


class InsertClassifier():

    def __init__(self):

        #super().__init__(InsertNet)
        self.headModel = InsertNet()

        self.DATA_DIR = config.DATA_DIR
        self.TRAIN_DIR = config.TRAIN_DIR
        self.VAL_DIR =  config.VAL_DIR

        self.finetuned_model_ckpt = config.finuted_model_ckpt_path
        self.scratch_model_ckpt = config.scratch_model_ckpt_path
        self.saveFinetunedModel = config.saveFinetunedModel
        self.saveScratchModel = config.saveScratchModel

        self.BATCH_SIZE = config.BATCH_SIZE
        self.IMG_SIZE = config.IMG_SIZE
        self.IMG_SHAPE = self.IMG_SIZE + (config.channels,)

        self.learning_rate = config.learning_rate
        self.initial_epochs = config.initial_epochs
        self.fine_tune_epochs = config.fine_tune_epochs
        self.class_mode = config.class_mode
        self.num_classes = config.num_classes

        self.num_folds = config.num_folds
        self.focal_loss = config.focal_loss

        self.imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32")

    def load_data(self):

        """
        Prepare train and validation data for training
        """

        # ----------------------------------------------------------------------------
        # Apply standard augmentation on train data
        # ----------------------------------------------------------------------------
        # initialize the training data augmentation object
        self.trainAug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.05,  # 0.15
            width_shift_range=0.1,  # 0.2
            height_shift_range=0.1,  # 0.2
            shear_range=0.05,  # 0.15
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )
        # initialize the validation (and testing) data augmentation object
        self.valAug = tf.keras.preprocessing.image.ImageDataGenerator()

        # ----------------------------------------------------------------------------
        # Substract imagenet mean values from images
        # ----------------------------------------------------------------------------

        self.trainAug.mean = self.imagenet_mean
        self.valAug.mean = self.imagenet_mean

        # ----------------------------------------------------------------------------
        # Load data from directory
        # ----------------------------------------------------------------------------
        # initialize the training generator
        self.trainGen = self.trainAug.flow_from_directory(
            self.TRAIN_DIR,
            class_mode=self.class_mode,
            target_size=self.IMG_SIZE,
            color_mode="rgb",
            shuffle=True,
            batch_size=self.BATCH_SIZE
        )
        # initialize the validation generator
        self.valGen = self.valAug.flow_from_directory(
            self.VAL_DIR,
            class_mode=self.class_mode,
            target_size=self.IMG_SIZE,
            color_mode="rgb",
            shuffle=True,
            batch_size=self.BATCH_SIZE
        )
        # ----------------------------------------------------------------------------

    def finetune(self):

        """Finetuning ResNet50 with simple_model"""

        # ----------------------------------------------------------------------------
        # TODO: Prefetching data into memory
        # ----------------------------------------------------------------------------
        #AUTOTUNE = tf.data.AUTOTUNE
        #self.trainGen = self.trainGen.prefetch(buffer_size=AUTOTUNE)
        #self.valGen = self.valGen.prefetch(buffer_size=AUTOTUNE)

        # ----------------------------------------------------------------------------
        # Call pretrained model as base model and set it as frozen
        # ----------------------------------------------------------------------------
        self.base_model = tf.keras.applications.ResNet50(  input_shape=self.IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')

        self.base_model.trainable = False
        self.base_model.summary()
        # ----------------------------------------------------------------------------
        # Define a simple model
        # ----------------------------------------------------------------------------
        inputs = tf.keras.Input(shape=self.IMG_SHAPE)
        #x = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        #x = tf.cast(x, tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = self.base_model(x, training=False) # x --> inputs
        outputs = self.headModel.simple_model(inputs=x)
        self.model = tf.keras.Model(inputs, outputs)

        # ----------------------------------------------------------------------------
        # Compile model and  check initial loss & accuracy
        # Finally train only headModel on top of frozen layers
        # ----------------------------------------------------------------------------
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr= self.learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

        loss0, accuracy0 = self.model.evaluate(self.valGen)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        self.history_base = self.model.fit(self.trainGen,
                                    epochs=self.initial_epochs,
                                    validation_data=self.valGen)

        # ----------------------------------------------------------------------------
        # Plot train/val accuracy for initial training
        # ----------------------------------------------------------------------------
        acc = self.history_base.history['accuracy']
        val_acc = self.history_base.history['val_accuracy']
        loss = self.history_base.history['loss']
        val_loss = self.history_base.history['val_loss']

        plot_accuracy_loss_curve(acc, val_acc , loss, val_loss, figure_save_path=config.history_base)
        # ----------------------------------------------------------------------------
        # Set pretrained model unfreeze and train from certain convolution layers
        # ----------------------------------------------------------------------------
        self.base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        if self.focal_loss:
            self.model.compile(loss=BinaryFocalLoss(gamma=2),
                               optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate / 10),
                               metrics=['accuracy'])

        else:
            self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate / 10),
                          metrics=['accuracy'])

        self.model.summary()
        # ----------------------------------------------------------------------------
        # Create a callback that saves the model's weights
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
                              filepath=self.finetuned_model_ckpt,
                              save_weights_only=True,
                              verbose=1,
                              period=4
                            )
        # ----------------------------------------------------------------------------
        total_epochs = self.initial_epochs + self.fine_tune_epochs

        self.history_finetuned = self.model.fit(self.trainGen,
                                                epochs=total_epochs,
                                                initial_epoch=self.history_base.epoch[-1],
                                                validation_data=self.valGen,
                                                callbacks=[self.cp_callback])

        # ----------------------------------------------------------------------------
        # Save entire model with weights and graph
        # ----------------------------------------------------------------------------

        self.model.save(self.saveFinetunedModel)

        # ----------------------------------------------------------------------------
        # Plot train/val accuracy for initial training
        # ----------------------------------------------------------------------------
        acc += self.history_finetuned.history['accuracy']
        val_acc += self.history_finetuned.history['val_accuracy']
        loss += self.history_finetuned.history['loss']
        val_loss += self.history_finetuned.history['val_loss']

        plot_accuracy_loss_curve(acc, val_acc , loss, val_loss, figure_save_path=config.history_finetuned)
        # ----------------------------------------------------------------------------

    def train_from_scratch(self):

        """Building model from scratch and train"""

        # ----------------------------------------------------------------------------
        # Calculate class weights
        # ----------------------------------------------------------------------------

        class_weights = calculate_class_weights(trainY=self.trainGen.classes)

        # ----------------------------------------------------------------------------
        # Get complex model to train from scratch and compile
        # ----------------------------------------------------------------------------
        model = self.headModel.complex_model(
                                               img_size=self.IMG_SIZE,
                                               channels=config.channels,
                                               num_classes=config.num_classes,
                                               last_activation=config.last_activation
                                            )
        model.summary()

        optimizer = Adagrad(lr=self.learning_rate, decay=self.learning_rate / self.initial_epochs)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      optimizer=optimizer,
                      metrics=["accuracy"])
        # ----------------------------------------------------------------------------
        # Create a callback that saves the model's weights
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=self.scratch_model_ckpt,
                                save_weights_only=True,
                                verbose=1,
                                period=4
                                )
        # ----------------------------------------------------------------------------
        STEP_SIZE_TRAIN = self.trainGen.n // self.trainGen.batch_size
        STEP_SIZE_VALID = self.valGen.n // self.valGen.batch_size

        self.history_scratch_model = model.fit(
                                                x=self.trainGen,
                                                steps_per_epoch=STEP_SIZE_TRAIN,
                                                validation_data=self.valGen,
                                                validation_steps=STEP_SIZE_VALID,
                                                class_weight=class_weights,
                                                epochs=self.initial_epochs
                                              )

        # ----------------------------------------------------------------------------
        # Save entire model and plot accuracy/loss curves
        # ----------------------------------------------------------------------------

        self.model.save(self.saveScratchModel)

        # ----------------------------------------------------------------------------
        # Plot train/val accuracy for initial training
        # ----------------------------------------------------------------------------
        acc = self.history_scratch_model.history['accuracy']
        val_acc = self.history_scratch_model.history['val_accuracy']
        loss = self.history_scratch_model.history['loss']
        val_loss = self.history_scratch_model.history['val_loss']

        plot_accuracy_loss_curve(acc, val_acc , loss, val_loss, figure_save_path=config.history_scratch_model)
        # ----------------------------------------------------------------------------


if __name__ == "__main__":

    classifier = InsertClassifier()
    classifier.load_data()

    if args['approach'] == 'finetune':
        classifier.finetune()
    elif args['approach'] == 'scratch':
        classifier.train_from_scratch()
    else:
        "Please pass a training approach to initiate the training"
