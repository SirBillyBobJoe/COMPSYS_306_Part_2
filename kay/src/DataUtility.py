import os
import pickle
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
from PIL import Image

def load_data_into_dataframe():
  # Path for locally saved kaggle dataset.
  datadir='kay/dataset'
  flat_data_arr=[]
  target_arr=[]
  # Get our array categories [0, ..., 42]
  signs = [str(i) for i in range(5)]

  for root, dirs, files in os.walk(datadir):
    for file in files:
        if file == '.DS_Store':
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Removed: {file_path}")

  for i in signs:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
      img_array = imread(os.path.join(path,img))
      img_resized = resize(img_array,(32,32,3))
      flat_data_arr.append(img_resized.flatten())
      target_arr.append(signs.index(i))
    print(f'loaded category:{i} successfully')
  flat_data = np.array(flat_data_arr)
  target = np.array(target_arr)
  df = pd.DataFrame(flat_data)
  df['Target']=target
  
  return df

# Save our dataframe so we don't need to load everytime.
df = load_data_into_dataframe()
pickle.dump(df, open('data.pickle', 'wb'))



# def resize_image(image_path, output_size=(32, 32)):
#     """Resize an image to the specified output size."""
#     try:
#         with Image.open(image_path) as img:
#             img = img.resize(output_size, Image.LANCZOS)
#             img.save(image_path)  # Overwrite the original image
#             print(f"Resized: {image_path}")
#     except Exception as e:
#         print(f"Failed to process {image_path}: {e}")

# def process_directory(directory, output_size=(32, 32)):
#     """Process all images within a directory and its subdirectories."""
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#                 image_path = os.path.join(root, file)
#                 resize_image(image_path, output_size)

# # Define the directory you want to process
# input_directory = 'kay/dataset'

# # Call the function
# process_directory(input_directory)
