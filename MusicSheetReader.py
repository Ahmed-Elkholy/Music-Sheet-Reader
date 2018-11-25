from commonfunctions import *
%matplotlib inline
%load_ext autoreload
%autoreload 2
from skimage import feature
import numpy as np
import cv2
import math
from scipy import ndimage
import operator
%config IPCompleter.greedy=True

#cropping (kholy's part)
#work in img_rotated
#img.minval,maxval,kernel_size
#img_rotated=cv2.GaussianBlur(img_rotated,(11,11),0)

def find_page_contours(img):
    img2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def find_largest_contour(contours):
    contours_area_list = []
    for contour in contours:
        contours_area_list.append(cv2.contourArea(contour))
        
    index, value = max(enumerate(contours_area_list), key=operator.itemgetter(1))
    #print(value)
    return contours[index]


def findApprox(contours):
    ff = []
    approx_list = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4):
            maxArea = cv2.contourArea(approx)
            ff.append(maxArea)
            approx_list.append(approx) 

    index = findMax(ff)        
    return approx_list[index]     



def findMax(ff):
    index, value = max(enumerate(ff), key=operator.itemgetter(1))

    return index

     
def resize(img, height=800):
    """ Resize image to given height """
    if (img.shape[0] > height):
        ratio = height / img.shape[0]
        return cv2.resize(img, (int(ratio * img.shape[1]), height))    
    
    
def sortCornerPoints(points):
    #print(points.shape)
    sorted_points = np.zeros_like(points)
    sum_points = np.sum(points,axis=1)
    sorted_points[0] = points[np.argmin(sum_points)]
    sorted_points[2] = points[np.argmax(sum_points)]
    diff_points = np.diff(points,axis=1)
    #print(points)
    sorted_points[1] = points[np.argmin(diff_points)]
    sorted_points[3] = points[np.argmax(diff_points)]
    return sorted_points
            
    
    
def transform_image(img,points):
    w = max(np.linalg.norm(points[0]-points[1]),np.linalg.norm(points[2]-points[3]))
    h = max(np.linalg.norm(points[0]-points[3]),np.linalg.norm(points[1]-points[2]))
    dest_img = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    
    #print(dest_img)
    #print(points)
    
    dest_img = dest_img.astype(np.float32)
    points = points.astype(np.float32)

    Trans_Matrix = cv2.getPerspectiveTransform(points,dest_img)
    cropped_img = cv2.warpPerspective(img, Trans_Matrix, (int(w), int(h)))
    cv2.imwrite('output/gg.jpg',img)
    cv2.imwrite('output/cropped_edge.jpg',cropped_img)
    return cropped_img


def crop_image(path):
    img_org = cv2.imread(path)
    
    img_before = resize(cv2.imread(path))
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.bilateralFilter(img_gray, 9, 75, 75)
    img_edges = cv2.adaptiveThreshold(img_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    img_edges = cv2.medianBlur(img_edges, 11)
    img_edges = cv2.Canny(img_edges,200, 250)
    img_edges = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, np.ones((5, 11)))

    cv2.imwrite('output/2_edge.jpg',img_edges)

    pageContour = find_page_contours(img_edges)
    con = find_largest_contour(pageContour)
    epsilon = 0.01*cv2.arcLength(con,True)
    approx = cv2.approxPolyDP(con,epsilon,True)
    ff= findApprox(pageContour)


    corner_points = sortCornerPoints(ff[:,0])*(img_org.shape[0]/800)
    cv2.drawContours(img_before,[ff],0,(0,0,255),3)
    cv2.imwrite('output/after_contour.jpg',img_before)



    return transform_image(img_org,corner_points)
    



def segment_image(img):

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
            filename = 'output/cropped' + str(imgIndex) + '.jpg'
            cv2.imwrite(filename,crop_img)
            cropHeight = 0
            imgIndex+=1
    return cropped_imgs
	
	
def detect_ellipses(image):
    gray_image = rgb2gray(image)*255
    gray_image = (gray_image.astype(np.uint8)<128) * np.uint8(255)
    kernel = np.ones((3,3), np.uint8)
    gray_image = cv2.erode(gray_image, kernel, iterations=5)
    gray_image = cv2.dilate(gray_image, kernel, iterations=1)
    return gray_image>128



#Main
img = crop_image('imgs/2.jpg')
segments = segment_image(img)
imgIndex = 1 
for x in segments:
    ellipses = detect_ellipses(x)
    x[ellipses] = (255, 0, 0)
    show_images([x])
    filename = 'output/ellipses' + str(imgIndex) + '.jpg'
    cv2.imwrite(filename,x)
    imgIndex+=1

	