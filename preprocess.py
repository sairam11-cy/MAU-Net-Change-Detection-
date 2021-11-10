import cv2
import numpy as np
import random
import math
import os
def randomCrop(img,img1, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img1.shape[0] >= height
    assert img1.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    assert img1.shape[0] == mask.shape[0]
    assert img1.shape[1] == mask.shape[1]
    imgs=[]
    imgs1=[]
    masks=[]
    checkx=[]
    checky=[]
    total_w=np.sum(mask == 255)
    total_w=total_w/(1500*1500)
    n_white_pix = np.sum(mask == 255)
    n_black_pix = np.sum(mask == 0)
    total=(math.log(n_black_pix)/math.log(n_white_pix))*0.1
    print(total)
    for i in range(500):
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        if(x in checkx or y in checky):
            continue
        checkx.append(x)
        checky.append(y)
        timg = img[y:y+height, x:x+width]
        timg1 = img1[y:y+height, x:x+width]
        tmask = mask[y:y+height, x:x+width]
        n_white_pix = np.sum(tmask >= 100)
        if(n_white_pix<10):
            continue
        imgs.append(timg)
        imgs1.append(timg1)
        masks.append(tmask)
    return imgs,imgs1, masks
filelist=os.listdir('train/img1/')
for i in range(len(filelist)):
    print(i)
    img11=cv2.imread("./train/img1/"+filelist[i])
    img12=cv2.imread("./train/img2/"+filelist[i])
    file=filelist[i]
    mask=cv2.imread("./train/mask/"+file,0)
    print(img11.shape)
    print(mask.shape)
    img1,img2,mask1=randomCrop(img11,img12, mask, 64, 64)
    img1=np.array(img1)
    img2=np.array(img2)
    mask1=np.array(mask1)
#cv2.imwrite("sample.png",img1)
#cv2.imwrite("sample2.png",mask1)
    print(img1.shape,mask1.shape)
    for j in range(len(img1)):
        cv2.imwrite("dataset/img1/"+str(i)+str(j)+".png",img1[j])
        cv2.imwrite("dataset/img2/"+str(i)+str(j)+".png",img2[j])
        cv2.imwrite("dataset/mask/"+str(i)+str(j)+".png",mask1[j])
'''
n_white_pix = np.sum(mask == 255)
n_black_pix = np.sum(mask == 0)
print(n_white_pix)
print(n_black_pix)
print(n_black_pix/(1500*1500),n_white_pix/(1500*1500))
total=((n_black_pix/1500*1500)/(n_white_pix/1500*1500))*0.001
print(total)
'''
