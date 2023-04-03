import os
import tensorflow as tf
import matplotlib.pyplot as plt
from create_datasets import create_datasets
from create_model import create_model
from utils.read_write_array import *
tf.get_logger().setLevel('ERROR')

# https://towardsdatascience.com/recurrent-neural-networks-explained-with-a-real-life-example-and-python-code-e8403a45f5de
# https://keras.io/examples/nlp/active_learning_review_classification/

NEW_MODEL = False

train_dataset, test_dataset = create_datasets()

if NEW_MODEL:
  model = create_model(train_dataset, test_dataset)
  model.summary()
else:
  model = tf.keras.models.load_model('./model')

# checkpoint_filepath = './tmp'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_freq=25,
#   )


# fit the model
history = model.fit(
    train_dataset,
    epochs=25,
    steps_per_epoch = len(train_dataset),
    validation_data=test_dataset,
    validation_steps=10,
    # callbacks=[model_checkpoint_callback]  
  )


# model evaluation
test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
# Visualize Model Loss and Accuracy

if NEW_MODEL:
  write_arr(history.history['mse'], './history/mse_history.csv')
  write_arr(history.history['loss'], './history/loss_history.csv')
else:
  append_arr(history.history['mse'], './history/mse_history.csv')
  append_arr(history.history['loss'], './history/loss_history.csv')

def plot_graphs(arr, metric):
  plt.plot(arr)
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric])

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(read_arr('./history/mse_history.csv'), 'mse')
plt.subplot(1, 2, 2)
plot_graphs(read_arr('./history/loss_history.csv'), 'loss')

plt.savefig("train")

model.save('./model')