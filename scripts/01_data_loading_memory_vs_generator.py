#!/usr/bin/env python3

import asyncio


import os
import matplotlib.pyplot as plt
import skillsnetwork

from PIL import Image

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"

extraction_path = "."
asyncio.run(skillsnetwork.prepare(url = url, path = extraction_path, overwrite = True))

# Define directories
extract_dir = "."

base_dir = os.path.join(extract_dir, 'images_dataSAT')
dir_non_agri = os.path.join(base_dir, 'class_0_non_agri')
dir_agri = os.path.join(base_dir, 'class_1_agri')

non_agri = os.scandir(dir_non_agri)
# print first 5 file paths
for f_path in range(5):
    print(next(non_agri))

file_name = next(non_agri)
print(file_name)
print(os.path.isfile(file_name))

image_data = plt.imread(file_name.path)


print(image_data.shape)

plt.imshow(image_data)


non_agri_images = os.listdir(dir_non_agri)
# print first 5 file paths
non_agri_images[:5]

non_agri_images.sort()

# print first 5 file paths
non_agri_images[:5]

image_data = Image.open(os.path.join(dir_non_agri, non_agri_images[0]))

image_data

plt.imshow(image_data)

non_agri_images_paths = [os.path.join(dir_non_agri, image) for image in non_agri_images]
#print first five paths
non_agri_images_paths[:5]

len(non_agri_images_paths)

n_images = 4
for image_path in non_agri_images[:n_images]:
    print(image_path)
    image_data = Image.open(os.path.join(dir_non_agri, image_path))
    plt.imshow(image_data)
    plt.show()


agri_images_paths = []
for image in os.listdir(dir_agri):
    agri_images_paths.append(os.path.join(dir_agri, image))

agri_images_paths.sort()


print(len(agri_images_paths))


n_images = 4
for image_path in agri_images_paths[:n_images]:
    print(image_path)
    image_data = Image.open(image_path)
    plt.imshow(image_data)
    plt.show()
