import os
from os import path
import random

BASE_DIR = os.getcwd()
TAGGED_DATA = BASE_DIR + '/tagged_data'


def load_classification():  # returns classification of mammogram
    classification_lines = open(path.join(BASE_DIR, 'image_classification.txt')).readlines()
    classification = {}
    for classification_line in classification_lines:
        classification_parts = [x.strip() for x in classification_line.split()]
        classification[classification_parts[0]] = classification_parts[2]
    return classification


def create_tagged_dataset():
    classification_map = load_classification()
    data_path = path.join(BASE_DIR, 'raw_data')
    raw_files = os.listdir(data_path)
    for raw_file in raw_files:
        print(f'Processing file {raw_file}')
        if not os.path.isfile(path.join(data_path, raw_file)) or not raw_file.endswith('.png'):
            print(f'Skipping file {raw_file}')
            continue
        filename, extension = os.path.splitext(raw_file)
        classification = classification_map[filename]
        print(f'Mapping {raw_file} to {classification}\n')

        random_percentage = random.randint(0, 100)

        if random_percentage <= 15:  # Fifteen percent change that the data is moved to the test directory
            put_into_dir("test", raw_file, classification)
        else:
            put_into_dir("train", raw_file, classification)


def put_into_dir(specified_dir, raw_file, classification):
    print(f"Moving data to {specified_dir} dir")

    os.rename(f"{BASE_DIR}/raw_data/{raw_file}", f"{path.join(TAGGED_DATA, path.join(specified_dir), classification, raw_file)}")


if __name__ == '__main__':
    create_tagged_dataset()
