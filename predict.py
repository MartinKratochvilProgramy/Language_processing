import tensorflow as tf
import os
from tabulate import tabulate

model = tf.keras.models.load_model('./model')

def parse_prediction(input_string):
    return round(model.predict([input_string], verbose=0)[0][0], 3)

def predict_folder(path: str):
    dir_list = os.listdir(path)

    sentiment = "negative" if path.split("/")[-1] == "class_negative" else "positive"

    results = []
    for file in dir_list:
        file_path = os.path.join(path, file)
        f = open(file_path, "r")  
        
        results.append([sentiment, parse_prediction(f.read()), file])
        
        f.close()  
    print("\n")
    print(tabulate(results, headers=["Sentiment", "Rating", "File name"], tablefmt='orgtbl'))

predict_folder('./datasets/test/class_negative')
predict_folder('./datasets/test/class_positive')