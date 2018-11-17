from commonfunctions import *

def segment_image(img):

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150)
    h_img,w_img,x = img.shape
    thrsh = 120
    img_edges = cv2.resize(img_edges, (w_img, h_img)) 
    img_edges[img_edges>=thrsh] = 255
    img_edges[img_edges<thrsh] = 0
    kernel = np.ones((50,50), np.uint8)
    dilated_img = cv2.dilate(img_edges, kernel, iterations=1)
    h,w = dilated_img.shape
    flag = 0
    cropHeight = 0
    imgIndex = 0
    cropped_imgs = list()
    bound = int(h/45)

    for y in range(h):
        if dilated_img[y,int(w/2)] == 255 and flag == 0:
            y_start = y
            flag = 1
        if dilated_img[y,int(w/2)] == 255 and flag == 1:
            cropHeight +=1
        if dilated_img[y,int(w/2)] == 0 and flag == 1:
            flag = 0  
            if cropHeight < 100:
                continue
            if imgIndex == 0:
                imgIndex+=1
                continue
            crop_img = img[y_start-bound:y_start+cropHeight+bound, :]
            cropped_imgs.append(crop_img)
            filename = 'cropped' + str(imgIndex) + '.jpg'
            cv2.imwrite(filename,crop_img)
            cropHeight = 0
            imgIndex+=1
    return cropped_imgs

img = cv2.imread('imgs/4.jpg')
segments = segment_image(img)
for x in segments:
    show_images([x])