from PIL import Image, ImageEnhance
import numpy as np
import os
from os import listdir

path = "./test/labelme"
files = listdir(path)
# print(files)
# filename = []
# for f in files:
#     filename.append(f)

for i in files:
    img = Image.open('./test/labelme/{}/label.png'.format(i))
    img = Image.fromarray(np.uint8(img))
    i = i.replace("_json","")
    # img = Image.fromarray(np.uint8(img)*20)
    # img.show()
    img.save('./test/mask/{}.png'.format(i))


