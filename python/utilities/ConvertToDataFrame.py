import os
import numpy as np
from skimage.io import imread
import pickle
from skimage.transform import resize

datadir = "./validation"
label_csv = "./labels.csv"
flat_data_arr = []
target_arr = []

# Read the CSV file manually, avoiding pandas
labels_df = []
with open(label_csv, 'r') as f:
    for line in f.readlines()[1:]:  # Skipping header
        class_id = line.strip().split(",")[0]
        labels_df.append(int(class_id))

print(f"Loaded labels:\n{labels_df[:5]}")

# Iterate over the labels and load images
for class_id in labels_df:
    path = os.path.join(datadir, str(class_id))
    for img in os.listdir(path):
        try:
            img_array = imread(os.path.join(path, img))
            resized_img = resize(img_array, (32, 32), anti_aliasing=True)
            flattened_img = resized_img.flatten()

            # Ensure consistent shape
            if len(flat_data_arr) > 0 and len(flattened_img) != len(flat_data_arr[0]):
                print(f"Skipping image {img} due to inconsistent shape: {flattened_img.shape}")
                continue

            flat_data_arr.append(flattened_img)
            target_arr.append(class_id)
        except Exception as e:
            print(f"Error processing {img}: {e}")
            continue

    print(f"Loaded category: {class_id} successfully")

# Convert to numpy arrays
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Combine data manually, mimicking the pandas dataframe structure
combined_data = {
    "data": flat_data,
    "target": target
}

# Print shape of data for confirmation
print("Data shape:", flat_data.shape)
print("Target shape:", target.shape)

# Dump the data using pickle
pickle.dump(combined_data, open("./pickles/validationData.pickle", "wb"))
print("Pickle is dumped successfully")
