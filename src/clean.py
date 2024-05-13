# Images
from genericpath import isfile
from os import listdir
from os.path import isfile, join
import os


imgs = [f for f in listdir("data/dirty_sink") if isfile(join("data/dirty_sink", f))]
print(imgs)

num = 0

extensions = [".jpg", ".webp", ".png"]

for ext in extensions:
    for img in imgs:
        if img.endswith(ext):
            os.rename(
                "data/dirty_sink/" + img, "data/dirty_sink/" + "dirty-" + str(num) + ext
            )
            num += 1
