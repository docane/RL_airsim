import os
import h5py
from Generator import DriveDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping

batch_size = 32
learning_rate = 0.0001
number_of_epochs = 500
activation = 'relu'
out_activation = 'tanh'
training_patience = 20
COOKED_DATA_DIR = './cooked_data/'
MODEL_OUTPUT_DIR = 'models_2023_02_13_13_53_30/'
train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
num_train_examples = train_dataset['image'].shape[0]
num_eval_examples = eval_dataset['image'].shape[0]
data_generator = DriveDataGenerator(rescale=1. / 255., horizontal_flip=False)
train_generator = data_generator.flow(train_dataset['image'],
                                      train_dataset['previous_state'],
                                      train_dataset['label'],
                                      batch_size=batch_size,
                                      zero_drop_percentage=0.95,
                                      roi=[78, 144, 27, 227])
eval_generator = data_generator.flow(eval_dataset['image'],
                                     eval_dataset['previous_state'],
                                     eval_dataset['label'],
                                     batch_size=batch_size,
                                     zero_drop_percentage=0.95,
                                     roi=[78, 144, 27, 227])
[sample_batch_train_data, sample_batch_test_data] = next(train_generator)

image_input_shape = sample_batch_train_data[0].shape[1:]

pic_input = Input(shape=image_input_shape)
img_stack = Conv2D(32, 8, name="conv1", strides=4, padding="valid", activation=activation)(pic_input)
img_stack = Conv2D(64, 4, name="conv2", strides=2, padding="valid", activation=activation)(img_stack)
img_stack = Conv2D(64, 3, name="conv3", strides=1, padding="valid", activation=activation)(img_stack)
img_stack = Flatten(name='flatten')(img_stack)
img_stack = Dense(256, name="fc1", activation=activation)(img_stack)
img_stack = Dense(128, name="fc2", activation=activation)(img_stack)
img_stack = Dense(64, name="fc3", activation=activation)(img_stack)
img_stack = Dense(1, name="output", activation=out_activation)(img_stack)

adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = Model(inputs=[pic_input], outputs=img_stack)
model.compile(optimizer=adam, loss='mse')
model.summary()

plateau_callback = ReduceLROnPlateau(monitor="val_loss",
                                     factor=0.5,
                                     patience=3,
                                     min_lr=learning_rate,
                                     verbose=1)

# csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))

checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR,
                                   'fresh_models',
                                   '{0}_model.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))

checkpoint_callback = ModelCheckpoint(checkpoint_filepath,
                                      save_best_only=True,
                                      verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_loss",
                                        patience=training_patience,
                                        verbose=1)

# callbacks = [plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]
callbacks = [plateau_callback, checkpoint_callback, early_stopping_callback]
history = model.fit_generator(train_generator,
                              steps_per_epoch=num_train_examples // batch_size,
                              epochs=number_of_epochs,
                              callbacks=callbacks,
                              validation_data=eval_generator,
                              validation_steps=num_eval_examples // batch_size,
                              verbose=1)
