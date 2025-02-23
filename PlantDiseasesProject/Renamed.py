import os
import shutil

def merge_and_rename_images(original_folder, augmented_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    classes = [d for d in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, d))]

    for class_name in classes:
        original_class_dir = os.path.join(original_folder, class_name)
        augmented_class_dir = os.path.join(augmented_folder, class_name)
        output_class_dir = os.path.join(output_folder, class_name)

        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        all_images = []

        if os.path.exists(original_class_dir):
            all_images.extend([(original_class_dir, f) for f in os.listdir(original_class_dir)])
        if os.path.exists(augmented_class_dir):
            all_images.extend([(augmented_class_dir, f) for f in os.listdir(augmented_class_dir)])

        for idx, (folder, image_name) in enumerate(all_images):
            _, ext = os.path.splitext(image_name)
            new_name = f"{class_name}_{idx + 1:04d}{ext}"

            old_path = os.path.join(folder, image_name)
            new_path = os.path.join(output_class_dir, new_name)

            shutil.copy(old_path, new_path)
            print(f"Копирано и преименувано: {old_path} -> {new_path}")



original_path = "C:/Users/elena/PycharmProjects/PlantDiseasesProject/PlantDataset/training"
augmented_path = "C:/Users/elena/PycharmProjects/PlantDiseasesProject/PlantDataset/training_augmented"
output_path =  "C:/Users/elena/PycharmProjects/PlantDiseasesProject/PlantDataset/training_merged"

merge_and_rename_images(original_path, augmented_path, output_path)
