import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def augment_all_classes(input_base_dir, output_base_dir, target_count=1000):

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for class_name in os.listdir(input_base_dir):
        class_input_dir = os.path.join(input_base_dir, class_name)
        class_output_dir = os.path.join(output_base_dir, class_name)

        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        files = os.listdir(class_input_dir)
        current_count = len(files)

        print(f"Класа: {class_name}, Почетен број: {current_count}")

        if current_count >= target_count:
            print(f"Класата {class_name} веќе има доволно слики.")
            continue

        needed_images = target_count - current_count
        print(f"Недостигаат {needed_images} слики за класата {class_name}")


        generated_count = 0
        while generated_count < needed_images:
            file = np.random.choice(files)
            img = load_img(os.path.join(class_input_dir, file))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for batch in datagen.flow(x, batch_size=1, save_to_dir=class_output_dir, save_prefix='aug',
                                      save_format='jpg'):
                generated_count += 1
                if generated_count >= needed_images:
                    break

        print(f"Број на слики по аугментација за {class_name}: {current_count + generated_count}")

input_path =  "C:/Users/elena/PycharmProjects/PlantDiseasesProject/PlantDataset/training"
output_path = "C:/Users/elena/PycharmProjects/PlantDiseasesProject/PlantDataset/training_augmented"

augment_all_classes(input_path, output_path)

