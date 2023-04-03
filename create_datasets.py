import os
import tensorflow as tf
import numpy as np

def create_datasets():
    train_dir = os.path.join("", "./datasets/train")
    train_dataset = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        label_mode = "int",
        labels = "inferred",
        follow_links = True
    )

    test_dir = os.path.join("", "./datasets/test")
    test_dataset = tf.keras.utils.text_dataset_from_directory(
        test_dir,
        label_mode = "int",
        labels = "inferred",
        follow_links = True
    )

    return train_dataset, test_dataset

if __name__ =='__main__':
    train_dataset, test_dataset = create_datasets()

    print(train_dataset)
    exit()

    for i, label in enumerate(train_dataset.class_names):
        print("Label", i, "corresponds to", label)

    classes = [0, 0, 0, 0, 0]
          
    for text_batch, label_batch in train_dataset.take(2):
        for i in range(len(text_batch)):
            # print("Review", text_batch.numpy()[i])
            # print("Label", label_batch.numpy()[i])

            classes = np.add(classes, label_batch.numpy()[i])

    print(classes)
