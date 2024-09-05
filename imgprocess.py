from PIL import Image
import os

input_folder = 'downloaded_images'
output_folder = 'processed_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg','.png','jpeg')):
        img_path = os.path.join(input_folder,filename)
        with Image.open(img_path) as img:
            img_resize = img.resize((256,256),Image.LANCZOS)
            img_resize.save(os.path.join(output_folder,filename))