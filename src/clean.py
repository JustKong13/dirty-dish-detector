# Images
from genericpath import isfile
from os import listdir
from os.path import isfile, join
import os


imgs = [f for f in listdir("data/clean_sink") if isfile(join("data/clean_sink", f))]
print(imgs)

jpg = 0
webp = 0
png = 0

for img in imgs:
    # rewrite file name to be dirty-x
    if img.endswith(".jpg"):
        os.rename(
            "data/clean_sink/" + img, "data/clean_sink/" + "clean-" + jpg + ".jpg"
        )
        jpg += 1
    elif img.endswith(".webp"):
        os.rename(
            "data/clean_sink/" + img, "data/clean_sink/" + "clean-" + jpg + ".webp"
        )
        webp += 1
    elif img.endswith(".png"):
        os.rename(
            "data/clean_sink/" + img, "data/clean_sink/" + "clean-" + jpg + ".png"
        )
        png += 1
