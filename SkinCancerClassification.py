import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, SeparableConv2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix

width, height = 224, 224


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)


train_data = train_datagen.flow_from_directory(
    "/Users/silakebapci/Desktop/Skin Cancer/data/train",
    target_size=(width, height),
    batch_size=64,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    "/Users/silakebapci/Desktop/Skin Cancer/data/test",
    target_size=(width, height),
    batch_size=32,
    class_mode="binary"
)

class Inception(tf.keras.Model):
    def __init__(self, c1, c2, c3, c4):  
        super().__init__()
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same', activation='relu')
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same', activation='relu')
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
    MaxPooling2D(),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D(),
    SeparableConv2D(128, 3, activation='relu', padding='same'),
    SeparableConv2D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    SeparableConv2D(256, 3, activation='relu', padding='same'),
    SeparableConv2D(256, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    SeparableConv2D(512, 3, activation='relu', padding='same'),
    SeparableConv2D(512, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.7),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = LearningRateScheduler(exponential_decay_fn)
checkpoint_cb = ModelCheckpoint("SkinCancer_model.keras", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)


history = model.fit(
    train_data,
    validation_data=test_data,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=20
)


loss, accuracy = model.evaluate(test_data)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


vgg = VGG19(input_shape=(width, height, 3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False


x = Flatten()(vgg.output)


inception_block = Inception(64, (64, 128), (32, 64), 64)
x = tf.keras.layers.Reshape((7, 7, -1))(x)
x = inception_block(x)
x = Flatten()(x)


prediction = Dense(1, activation="sigmoid")(x)


model_vgg_inception = Model(inputs=vgg.input, outputs=prediction)
model_vgg_inception.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history_vgg_inception = model_vgg_inception.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
)


loss_vgg_inception, accuracy_vgg_inception = model_vgg_inception.evaluate(test_data)
print(f"VGG+Inception Model - Loss: {loss_vgg_inception}")
print(f"VGG+Inception Model - Accuracy: {accuracy_vgg_inception}")


y_true = test_data.classes
y_pred = (model_vgg_inception.predict(test_data) > 0.5).astype(int)

conf_matrix = confusion_matrix(y_true, y_pred)
tp = conf_matrix[1, 1]
tn = conf_matrix[0, 0]
fp = conf_matrix[0, 1]
fn = conf_matrix[1, 0]

sensitivity_conf = tp / (tp + fn)
specificity_conf = tn / (tn + fp)
accuracy_conf = (tp + tn) / (tp + tn + fp + fn)


history = history_vgg_inception.history
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs_range = range(len(accuracy))

plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label="Training Accuracy")
plt.plot(epochs_range, val_accuracy, label="Test Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Test Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Test Loss")
plt.legend(loc="upper right")
plt.title("Training and Test Loss")

plt.figtext(0.5, 0.95, f"Accuracy: {accuracy_conf*100:.2f}% | Sensitivity: {sensitivity_conf*100:.2f}% | Specificity: {specificity_conf*100:.2f}%", ha="center", fontsize=12)

plt.show()


test_images, test_labels = next(test_data)
num_images_to_show = 10

plt.figure(figsize=(20, 8))
for i in range(num_images_to_show):
    img = test_images[i]
    label = test_labels[i]
    
    img_array = np.expand_dims(img, axis=0)
    prediction = model_vgg_inception.predict(img_array)
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    
    actual_label = "Benign" if label == 0 else "Malignant"
    predicted_label = "Benign" if prediction[0][0] < 0.5 else "Malignant"
    plt.title(f"Real: {actual_label}\nGuess: {predicted_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
