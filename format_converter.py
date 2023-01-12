import os
from PIL import Image


def convert_to_png():
    path_specified = os.getcwd()
    mammograms_raw_path = f"{path_specified}/mammograms_raw"
    print(f"Current Working Directory: {mammograms_raw_path}")
    for file in os.listdir(f"{mammograms_raw_path}"):
        filename, extension = os.path.splitext(file)

        print(f"File: {file}")
        print(f"Converting file {file} to {filename}.png")

        if extension == ".pgm":
            print(f"Filename: {filename}.png")
            new_file = f"{filename}.png"
            print(f"new file name: {new_file}")
            with Image.open(f"{path_specified}/mammograms_raw/{file}") as im:
                im.save(f"{path_specified}/mammograms_raw/{new_file}")


if __name__ == '__main__':
    convert_to_png()
