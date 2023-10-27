import os
import random
import glob
from os import getcwd
import cv2

current_dir = getcwd()
cnt = 0
IMG_HEIGHT = 480
IMG_WIDTH = 640

####Process
list = []
for f_path in glob.iglob(os.path.join(current_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(f_path))
        list.append(title)

	
while list:
        name = random.choice(list)
        print(name)
        tmp = cv2.imread((current_dir.replace('\\','/')+'/'+name+'.jpg'))

        f = open(name+".txt", "r")
        txtfile = f.readlines()

        cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        
        for line in txtfile :
        #print(contents)
        
            contents = line.split()
            f.close()
    
            start_x = (float(contents[1])-float(contents[3])/2) * IMG_WIDTH
            start_y = (float(contents[2])-float(contents[4])/2) * IMG_HEIGHT

            end_x = (float(contents[1])+float(contents[3])/2) * IMG_WIDTH
            end_y = (float(contents[2])+float(contents[4])/2) * IMG_HEIGHT
        
            cv2.rectangle(tmp, (int(start_x),int(start_y)), (int(end_x), int(end_y)), (0,0,255), 2)
            cv2.putText(tmp, contents[0], (int(start_x),int(start_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        cv2.imshow("image", tmp);
        cv2.resizeWindow("image", 1280, 960)
        cv2.waitKey(0)
        list.remove(name)
