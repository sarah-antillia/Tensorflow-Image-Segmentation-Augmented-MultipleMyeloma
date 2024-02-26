import os
import glob
import shutil
import sys

from PIL import Image

def bmp2jpg(input_dir, output_dir):
  filepaths = glob.glob(input_dir + "/*.bmp")
  for filepath in filepaths:
    img = Image.open(filepath)
    basename = os.path.basename(filepath)
    nameonly = basename.split(".")[0]
    output_filepath = os.path.join(output_dir, nameonly + ".jpg")
    img.save(output_filepath, 'JPEG' ,quality=95)
    print("=== Converted from {} to {}".format(filepath, output_filepath))

if __name__ == "__main__":
  input_dir  = "./mini_test_bmp"
  output_dir = "./mini_test"
  bmp2jpg(input_dir, output_dir)
