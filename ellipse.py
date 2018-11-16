from commonfunctions import *

# Ellipse detection (Aboadma)
def detect_ellipses(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.erode(image, kernel, iterations=7)
    image = cv2.dilate(image, kernel, iterations=1)
    return image>128

image = io.imread('imgs/8.jpg')
gray_image = rgb2gray(image)*255
gray_image = (gray_image.astype(np.uint8)<128) * np.uint8(255)
ellipses = detect_ellipses(gray_image)
image[ellipses] = (255, 0, 0)
io.imshow(image)
io.show()
