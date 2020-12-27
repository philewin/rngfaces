import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2
from datetime import datetime

rngtype = "prng"
p=1000
nimg=1
sinput=False
for n in range(nimg):

    ar=[]


    now = datetime.now().strftime("%m%d%Y%H%M%S")+"_p"+str(p)+"_"+rngtype
    print("generating img_input{0}.png".format(now))
    if rngtype == "urandom":
        for z in range(p):
            muh=[]
            for i in range(p):
                with open("/dev/urandom", 'rb') as f:
                    muh.append(int.from_bytes(f.read(1), "little"))
            ar.append(muh)
        arr = np.array(ar)
    elif rngtype == "trng":
        for z in range(p):
            muh=[]
            for i in range(p):
                with open("/dev/random", 'rb') as f:
                    muh.append(int.from_bytes(f.read(1), "little"))
            ar.append(muh)
        arr = np.array(ar)
    elif rngtype == "prng":
        arr = np.random.rand(p,p)
    arr = (arr / arr.max())*255

    _ = plt.imshow(arr, cmap='gray',interpolation='none')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("faces/imgwithfaces/raw/img_rawfaces{0}.png".format(now), transparent=True, bbox_inches="tight", pad_inches=0,   dpi=600)
    plt.cla()



    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread("faces/imgwithfaces/raw/img_rawfaces{0}.png".format(now))
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)#minSize=(100,100)
    flen=len(faces)
    if flen > 0:
        print("FOUND "+str(flen)+" FACES!")
        for idx, (x, y, w, h) in enumerate(faces):
            print("Face"+ str(idx+1))
            cv2.imwrite('faces/onlyfaces/face{0}.png'.format(str(now)+"_"+str(idx+1)), img[y:(y+h),x:(x+w)])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 10)
    # Display the output
        cv2.imwrite('faces/imgwithfaces/rectangles/faceimg{0}.png'.format(now), img)
    else:
        print("no faces :-(")
        if not sinput:
            os.remove("faces/imgwithfaces/raw/img_rawfaces{0}.png".format(now))
        else:
            os.system('cp faces/imgwithfaces/raw/img_rawfaces{0}.png'.format(now)+" " +'nofaces/img_raw{0}.png'.format(now) )  
        


