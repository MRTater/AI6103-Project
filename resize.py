import os

from PIL import Image

DATAFOLDER_PATH = 'data/images/images'
files = os.listdir(DATAFOLDER_PATH)


def resize(sample_size, img_size):
    for file in sorted(files[:sample_size]):
        original_file = os.path.join(DATAFOLDER_PATH, file)
        resized_file = os.path.join('data', 'resized', file)

        image = Image.open(original_file)

        resized_image = image.resize((img_size, img_size))
        resized_image.save(resized_file)


if __name__ == '__main__':
    if os.path.exists("data/resized"):
        print("resized exist, delete first")
    else:
        os.mkdir("data/resized")
    resize(50000, 256)
