import tensorflow as tf 
from clean_data import clean_dataset
from reduce_overfitting import get_data_augmentation
from evaluate import evaluate_model
from predict import predict_image

# Clean the dataset first
clean_dataset()

# The model learns from training data.

train_data = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    image_size=(224, 224),
    batch_size=32,
    validation_split = 0.2, # 20% of total data for validation
    subset = "training",
    seed = 42,
    label_mode = "binary"
)

# The model never trains on validation data, only gets tested on it.

val_data = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    image_size = (224, 224),
    batch_size = 32,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    label_mode = "binary"
)

# Data Augmentatiom

data_augmentation = get_data_augmentation()

#Build model 

model = tf.keras.Sequential([

    # Layer one: normalize pixels 0-225 -> 0 - 1
    tf.keras.layers.Input(shape = ((224, 224, 3))),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),


    # Layer Two 
    # Block - 1
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # find patterns like : edge, color
    tf.keras.layers.MaxPooling2D(),                      # Srink image into half (4x4 -> 2x2)

    # Block - 2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # find more complex patterns like : detects shapes, ears, eyes
    tf.keras.layers.MaxPooling2D(),

    # Block - 3
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # find more complex patterns like : detects faces, fur patterns
    tf.keras.layers.MaxPooling2D(),

    #Flatten -> squishes the 3D to 1D
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),           # Dropout to rreduce overfitting

    # Last Layer
    tf.keras.layers.Dense(128, activation = 'relu'),     # thinking layer -> think and combine feature given by flatten(in one line)
    tf.keras.layers.Dropout(0.3),                        # Dropout to rreduce overfitting
    tf.keras.layers.Dense(1, activation= 'sigmoid')      # one output neuron -> 0 or 1 (binary) -> Cat or Dog?
])

# Compile Model

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# Train Model

history = model.fit(
    train_data,
    validation_data = val_data,
    epochs = 10,
)

# Evaluate

evaluate_model(model, val_data)

# Prediction

predict_image(model, "PetImages/Cat/1.jpg")
predict_image(model, "PetImages/Dog/1.jpg")