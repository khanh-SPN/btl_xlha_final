import os
import shutil

def move_image(image_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    shutil.move(image_path, target_dir)
