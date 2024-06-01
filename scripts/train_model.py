import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
train_set = joblib.load("train_set.pkl")

num_classes = len(train_set.class_indices)

# Transfer Learning setup
IMG_SHAPE = (256, 256, 3)
base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

base_model.trainable = False

# Add classification layers
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation="softmax")(x)
new_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
new_model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

# Train the model
steps_per_epoch = train_set.samples // train_set.batch_size
new_model.fit(
    train_set,
    steps_per_epoch=steps_per_epoch,
    epochs=20
)

# Save model
new_model.save('best_model.h5')