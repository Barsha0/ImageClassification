import tensorflow as tf 

load_data = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    image_size=(224, 224),
    batch_size=32,
    validation_split = 0.2, # 20% of total data for validation
    subset = "training",
    seed = 42,
    label_mode = "binary"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    image_size = (224, 224),
    batch_size = 32,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    label_mode = "binary"
)

