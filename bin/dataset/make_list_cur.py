import os
import random
import glob
from os import getcwd


current_dir = getcwd()

file_train = open('target.txt', 'w')


####Process
list = []
for f_path in glob.iglob(os.path.join(current_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(f_path))
        list.append(title)

	
while list:
        name = random.choice(list)
        print(name)
        file_train.write("{0}/{1}.jpg\n".format(current_dir.replace('\\','/'),name))
        list.remove(name)

file_train.close()
