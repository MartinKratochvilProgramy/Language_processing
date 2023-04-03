import os
import shutil
import random

def train_split(src_path, target_path, fraction):
    # clear files in target_path
    # move 'fraction' files from src_path to target_path

    # clear target path
    dir_list = os.listdir(target_path)
    for file in dir_list:
        file_path = os.path.join(target_path, file)
        os.remove(file_path)
        print(f"Removed {file_path}")

    dir_list = os.listdir(src_path)
    random.shuffle(dir_list)

    split_index = int(len(dir_list) * fraction)

    test_set = dir_list[:split_index]

    for file in test_set:
        src = os.path.join(src_path + "/" + file)
        target = os.path.join(target_path + "/" + file)
        
        shutil.move(src, target)
        print(f"Moved {src} to {target}")