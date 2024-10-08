import os
import numpy as np
from skimage.io import imread
import pandas as pd
import pickle

datadir = "myData"
label_csv = "labels.csv"
flat_data_arr = []
target_arr = []

labels_df = pd.read_csv(label_csv)
print(f"Loaded labels:\n{labels_df.head()}")

for class_id in labels_df["ClassId"]:
    print(f'Loading category: {class_id} - {labels_df.loc[labels_df["ClassId"] == class_id, "Name"].values[0]}')
    path = os.path.join(datadir, str(class_id))
    for img in os.listdir(path):
        try:
            img_array = imread(os.path.join(path, img))
            flattened_img = img_array.flatten()

            if len(flat_data_arr) > 0 and len(flattened_img) != len(flat_data_arr[0]):
                print(f"Skipping image {img} due to inconsistent shape: {flattened_img.shape}")
                continue

            flat_data_arr.append(flattened_img)
            target_arr.append(class_id)
        except Exception as e:
            print(f"Error processing {img}: {e}")
            continue

    print(f"Loaded category: {class_id} successfully")

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)
df["Target"] = target


print("DataFrame shape:", df.shape)
print(df)

pickle.dump(df, open("./pickles/data.pickle", "wb"))
print("Pickle is dumped successfully")

