'''
NOTES:
-----

The first and last lines in test case "13.JPG" were different not due to miss classification of pitch
They were different because the function vertical_segmentation removes the first 2 symbols
In this test case the clef and the 2 4s were nearly joined so they were considered a single symbol
This led to the removal of the first note in the line
Conclusion: This is fine

Half notes when their ellipses are detected and in the case they are after the last line
If D4 they get classified as E4
Conclusion: This is very fine
'''

from commonfunctions import *
import operator
from scipy.signal import find_peaks
from joblib import dump, load
from classifier import *
import pysynth
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def find_page_contours(img):
    img2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_largest_contour(contours):
    contours_area_list = []
    for contour in contours:
        contours_area_list.append(cv2.contourArea(contour))
    index, value = max(enumerate(contours_area_list), key=operator.itemgetter(1))
    return contours[index]


def findApprox(contours):
    ff = []
    approx_list = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if len(approx) == 4:
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
    if img.shape[0] > height:
        ratio = height / img.shape[0]
        return cv2.resize(img, (int(ratio * img.shape[1]), height))    
    
    
def sortCornerPoints(points):
    sorted_points = np.zeros_like(points)
    sum_points = np.sum(points, axis=1)
    sorted_points[0] = points[np.argmin(sum_points)]
    sorted_points[2] = points[np.argmax(sum_points)]
    diff_points = np.diff(points, axis=1)
    sorted_points[1] = points[np.argmin(diff_points)]
    sorted_points[3] = points[np.argmax(diff_points)]
    return sorted_points           
    
    
def transform_image(img, points):
    w = max(np.linalg.norm(points[0]-points[1]), np.linalg.norm(points[2]-points[3]))
    h = max(np.linalg.norm(points[0]-points[3]), np.linalg.norm(points[1]-points[2]))
    dest_img = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dest_img = dest_img.astype(np.float32)
    points = points.astype(np.float32)
    trans_matrix = cv2.getPerspectiveTransform(points, dest_img)
    cropped_img = cv2.warpPerspective(img, trans_matrix, (int(w), int(h)))
    cv2.imwrite('output/gg.jpg', img)
    cv2.imwrite('output/cropped_edge.jpg', cropped_img)
    return cropped_img


def rotate_with_lines(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=700, maxLineGap=80)
    staff_lines_length = []
    angles = []

    '''
    if (lines is None or lines.shape[0]<5):
        img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=1200, maxLineGap=80)
        img = img_org '''

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            staff_lines_length.append(x2-x1)
            angles.append(angle)
    lines_indices = np.argsort(np.asarray(staff_lines_length))
    lines_indices = lines_indices[::-1]
    lines_indices = lines_indices[0:5]
    staff_lines = lines[lines_indices]
    x_start, y1, x_end, y2 = staff_lines[0, 0]
    median_angle = np.median(angles)
    # print(median_angle)
    cv2.imwrite('output/before_return.jpg', img)
    img_rotated = ndimage.rotate(img, median_angle)
    cv2.imwrite('output/return.jpg', img_rotated)
    return img_rotated, x_start, x_end


def crop_image(path):
    img_org = cv2.imread(path)
    img_before = resize(cv2.imread(path))
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    # img_edges = cv2.bilateralFilter(img_gray, 9, 75, 75)
    thresh = threshold_sauvola(img_gray, window_size=45)
    img_edges = (img_gray > thresh).astype(np.uint8)
    img_edges = cv2.erode(img_edges, np.ones((5, 5)))
    # print(img_edges)
    cv2.imwrite('output/32_edge.jpg', img_edges*255)
    # show_images([img_edges])
    img_edges = img_edges*255
    # img_edges = cv2.adaptiveThreshold(img_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    img_edges = cv2.medianBlur(img_edges, 11)
    cv2.imwrite('output/1_edge.jpg', img_edges)

    img_edges = cv2.Canny(img_edges, 200, 250)
    cv2.imwrite('output/3_edge.jpg', img_edges)

    #img_edges = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, np.ones((5, 11)))

    #cv2.imwrite('output/2_edge.jpg',img_edges)

    pageContour = find_page_contours(img_edges)
    con = find_largest_contour(pageContour)
    epsilon = 0.01*cv2.arcLength(con, True)
    approx = cv2.approxPolyDP(con, epsilon, True)
    ff = findApprox(pageContour)

    corner_points = sortCornerPoints(ff[:, 0])*(img_org.shape[0]/800)
    cv2.drawContours(img_before, [ff], 0, (0, 0, 255), 3)
    cv2.imwrite('output/after_contour.jpg', img_before)

    transformed_img = transform_image(img_org, corner_points)
    return rotate_with_lines(transformed_img)


def segment_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_min, x_max = get_page_limits(img_gray)
    img_edges = cv2.Canny(img_gray, 50, 150)
    h_img, w_img, x = img.shape
    thrsh = 120
    img_edges = cv2.resize(img_edges, (w_img, h_img))
    img_edges[img_edges >= thrsh] = 255
    img_edges[img_edges < thrsh] = 0
    kernel = np.ones((50, 50), np.uint8)
    dilated_img = cv2.dilate(img_edges, kernel, iterations=1)
    h, w = dilated_img.shape
    flag = 0
    cropHeight = 0
    imgIndex = 0
    cropped_imgs = list()
    bound = int(h/45)
    for y in range(h):
        if dilated_img[y, int(w/2)] == 255 and flag == 0:
            y_start = y
            flag = 1
        if dilated_img[y, int(w/2)] == 255 and flag == 1:
            cropHeight += 1
        if dilated_img[y, int(w/2)] == 0 and flag == 1:
            flag = 0
            if cropHeight < 100:
                continue
            if imgIndex == 0:
                imgIndex += 1
                cropHeight = 0
                continue
            crop_img = img[y_start-bound:y_start+cropHeight+bound, x_min:x_max]
            cropped_imgs.append(crop_img)
            filename = 'output/cropped' + str(imgIndex) + '.jpg'
            cv2.imwrite(filename, crop_img)
            cropHeight = 0
            imgIndex += 1
    return cropped_imgs


def detect_ellipses(image):
    ellipses = np.copy(image).astype(np.uint8)
    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]).astype(np.uint8)
    ellipses = cv2.dilate(ellipses, kernel, iterations=3)
    ellipses = cv2.erode(ellipses, kernel, iterations=3)
    ellipses = ellipses < 0.5
    return ellipses


def remove_lines_seg(img):
    # bar(np.arange(img.shape[0]), 80 - np.sum(img,axis=1))
    y = 80 - np.sum(img, axis=1)
    maxnum = np.max(y) - 3
    number_of_peaks = np.sum(y > maxnum)
    thickness = number_of_peaks//5                         
    img[y > maxnum] = 1
    img_modified = img.copy()
    img_modified = cv2.medianBlur(img_modified.astype(np.uint8), 5)
    h, w = img_modified.shape
    bound = 6
    for y in range(bound, h - bound):
        for x in range(w):
            if img_modified[y, x] == 1 and img_modified[y-bound, x] == 0 and img_modified[y+bound, x] == 0:
                img_modified[y, x] = 0
    return img_modified


def remove_lines_s(img, segnum):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 53, 4)
    bin_img = bin_img/255
    delta = 80
    cv2.imwrite('line_re'+str(segnum)+'.jpg', bin_img*255)
    for i in range(0, bin_img.shape[1], delta):
        bin_img[:, i:i+delta] = remove_lines_seg(bin_img[:, i:i+delta])
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(bin_img, kernel, iterations=2)
    closed_img = cv2.dilate(eroded_img, kernel, iterations=2)
    return closed_img


def remove_ellipses(ellipses, bin_img):
    x_bin = bin_img > 0.5
    x_bin[ellipses] = True
    x_bin = x_bin.astype(np.uint8)
    return x_bin


def detect_centers(img, ellipses):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[ellipses] = 255
    img_gray[img_gray != 255] = 0
    lbl = ndimage.label(img_gray)[0]
    numberOfEllipses = np.max(lbl)
    arr = list(range(1, numberOfEllipses+1))
    centers = ndimage.measurements.center_of_mass(img_gray, lbl, arr)
    centers = np.asarray(centers).astype(int)
    return centers


# Obsolete
def segment_symbols(line):
    line = line.astype(np.uint8)
    line = np.logical_not(line)
    line = line.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(line.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.asarray(contours)
    bounding_rect = []
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if (w*h) < 500:
            continue
        bounding_rect.append((int(x), int(y), int(w), int(h)))
    return bounding_rect
    

# Obsolete
def draw_bounding_rect(line, bounding_rect):
    for x, y, w, h in bounding_rect:
        line = cv2.rectangle(line, (int(x), int(y)), (int(x+w), int(y+h)), 0, 2)
    cv2.imwrite("test4.jpg", line*255)
    

# Obsolete
def draw_histogram(line, boundingrects, pitches):
    cv2.imwrite("histo.jpg", line*255)
    global countQuarter
    global countEighth
    # print(len(pitches))
    rectCount = 0
    index = 0
    result = []
    for rect in boundingrects:
        if rectCount == 0 or rectCount == 1 or rectCount == 2:
            rectCount += 1
            continue
        x, y, w, h = rect
        if w < 14:
            continue
        symbol = line[int(y):int(y+h), int(x):int(x+w)]
        if len(symbol) == 0:
            continue
        if np.count_nonzero(symbol) == 0:
            continue
        peaks, _ = find_peaks(np.sum(symbol, axis=0), height=50)
        # print(peaks)
        if len(peaks) > 1:
            # print("EIGHTH")
            countEighth += 1
            result.append((pitches[index], "EIGHTH"))
            index += 1
            result.append((pitches[index], "EIGHTH"))
        else:
            countQuarter += 1
            # print("QUARTER OR HALF")
            result.append((pitches[index], "QUARTER OR HALF"))
        # bar(np.arange(symbol.shape[1]), np.sum(symbol,axis=0))
        # print(index)
        # show_images([symbol])
        index += 1
    return result


def find_first_zero(vec, index):
    for i in range(index, len(vec)):
        if vec[i] == 0:
            return i
    return -1


def find_last_zero(vec, index):
    for i in range(index, len(vec)):
        while i < len(vec) and vec[i] == 0:
            i += 1
        return i
    return -1


def vertical_segmentation(line):
    cv2.imwrite('test_el_test.jpg', line*255)
    line[:, 0] = 0
    vertical_projection = np.sum(line, axis=0)
    index = 0
    bounding_rect = []
    while index < len(vertical_projection):
        index1 = find_first_zero(vertical_projection, index)
        index = find_last_zero(vertical_projection, index1)
        xstart = index
        index1 = find_first_zero(vertical_projection, index)
        xend = index1
        index = index1
        if xend == -1 or xstart == -1:
            break
        bounding_rect.append((xstart, xend))
    return bounding_rect


def draw_bounding(line, lineno, boundingrects, pitches):
    global countQuarter
    global countEighth
    global model
    pitches = np.asarray(pitches)
    count = 0
    pitches_array = []
    skip = 0
    for rect in boundingrects:
        xstart, xend = rect
        symbol = line[:, int(xstart):int(xend)]
        symbol = compress_height(symbol)
        # show_images([symbol])
        if len(symbol) == 0:
            continue
        if np.count_nonzero(symbol) == 0:
            continue
        if symbol.shape[1] < 14:
            continue
        # skip first two segments if first line and one elsewhere
        if lineno == 0:
            if skip < 2:
                skip += 1
                continue
        else:
            if skip < 1:
                skip += 1
                continue
        f = pitches[:, 0]
        f = f.astype(np.float)
        pitches_inside = pitches[np.logical_and(f > xstart, f < xend)]
        if pitches_inside.shape[0] == 0:
            continue
        if pitches_inside.shape[0] == 1:
            pitches_inside = pitches_inside[0]
            if pitches_inside[2] == "X":
                continue
            # Trigger learning module
            res = predict(model, symbol)
            if res == 0:
                # print("eighth")
                pitches_array.append((pitches_inside[2], 8))
            else:
                # print("quarter")
                pitches_array.append((pitches_inside[2], 4))
        else:
            # print("eighth")
            for pitch in pitches_inside:
                if pitch[2] == "X":
                    continue
                pitches_array.append((pitch[2], 8))
    return pitches_array


# Obsolete
def fill_ellipses(image):
    data = np.copy(image)
    # finds and number all disjoint white regions of the image
    is_white = data > 0.5
    labels, n = ndimage.measurements.label(is_white)

    # get a set of all the region ids which are on the edge - we should not fill these
    on_border = set(labels[:, 0]) | set(labels[:, -1]) | set(labels[0, :]) | set(labels[-1, :])

    for label in range(1, n+1):  # label 0 is all the black pixels
        if label not in on_border:
            # turn every pixel with that label to black
            data[labels == label] = 0
    return data


def compress_height(symbol):
    hp = np.sum(symbol, axis=1)
    # print(hp)
    firstindex = np.argwhere(hp > 4)[0]
    hp = hp[::-1]
    secondindex = np.argwhere(hp > 4)[0]
    secondindex = symbol.shape[0] - secondindex
    symbol_new = symbol[int(firstindex):int(secondindex), :]
    return symbol_new


def get_page_limits(img):
    bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    vp = np.sum(bin_img, axis=0)
    peaks_s, _ = find_peaks(vp, distance=img.shape[1]//2)
    bin_img2 = (img > 128).astype(np.uint8)
    vp = np.sum(bin_img2, axis=0)
    peaks_s2, _ = find_peaks(vp, distance=img.shape[1]//2)
    peak = min(peaks_s[0], peaks_s2[0])
    cv2.imwrite("output/bin_peaks.jpg", bin_img)
    return peak, img.shape[1]-50


def getLines(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.Canny(img_gray, 50, 150)
    img_horizontal = np.copy(img_bin)
    kernel = np.ones((3, 3), np.uint8)
    img_horizontal = cv2.dilate(img_horizontal, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    img_horizontal = cv2.erode(img_horizontal, kernel, iterations=2)
    img_horizontal = cv2.dilate(img_horizontal, kernel, iterations=16)
    kernel = np.ones((4, 2), np.uint8)
    img_horizontal = cv2.erode(img_horizontal, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    img_horizontal = cv2.dilate(img_horizontal, kernel, iterations=20)
    cv2.imwrite('output/Hor_Lines.jpg', img_horizontal)
    return img_horizontal


def getMidLine(col, linesImage):
    lines = []
    h, w = linesImage.shape
    col = col - 10
    for y in range(h):
        count = 0
        for x in range(col-10, col+10):
            if x >= w:
                break
            if linesImage[y, x] == 255:
                count += 1
            if count >= 10:
                lines.append(y)
                if len(lines) > 1 and y - lines[-2] < 15:
                    lines.remove(y)
                    break
    lines = np.asarray(lines)
    # print("Obtained ", len(lines), " lines")
    if len(lines) < 3:
        midLine = 129
        avgSpace = 16
    else:
        midLine = lines[2]
        avgSpace = np.average(lines[1:] - lines[:len(lines)-1])
    return midLine, int(avgSpace)


def detectPositions(linesImage, centers, flag):
    centers = centers[centers[:, 1].argsort()]
    pos = []
    for center in centers:
        midLine, gapSize = getMidLine(center[1], linesImage)
        diff = midLine - center[0]
        pitches = {
            range(int(-13*gapSize/4), int(-11*gapSize/4)): "c4",  # L6
            range(int(-11*gapSize/4), int(-9*gapSize/4)):  "d4",  # L5L6
            range(int(-9*gapSize/4), int(-7*gapSize/4)):   "e4",  # L5
            range(int(-7*gapSize/4), int(-5*gapSize/4)):   "f4",  # L4L5
            range(int(-5*gapSize/4), int(-3*gapSize/4)):   "g4",  # L4
            range(int(-3*gapSize/4), int(-gapSize/4)):     "a4",  # L3L4
            range(int(-gapSize/4), int(gapSize/4)):        "b4",  # L3
            range(int(gapSize/4), int(3*gapSize/4)):       "c5",  # L2L3
            range(int(3*gapSize/4), int(5*gapSize/4)):     "d5",  # L2
            range(int(5*gapSize/4), int(7*gapSize/4)):     "e5",  # L1L2
            range(int(7*gapSize/4), int(9*gapSize/4)):     "f5",  # L1
            range(int(9*gapSize/4), int(11*gapSize/4)):    "g5",  # L0L1
            range(int(11*gapSize/4), int(13*gapSize/4)):   "a5",  # L0
        }
        # print(diff,gapSize)
        if diff not in range(int(-13*gapSize/4), int(13*gapSize/4)):
            pos.append([center[1], center[0], "X"])
        else:
            for x in pitches:
                if diff in x:
                    pos.append([center[1], center[0], pitches[x]])
                    continue
    return pos


########################################################################################################################
#                                                        Main                                                          #
########################################################################################################################
# Initialize counters
countQuarter = 0
countEighth = 0
imgIndex = 1
segnum = 0
# Initialize lists
song = []
# Load test case and learning module
#img, x_start, x_end = crop_image('imgs/13.JPG')
img, x_start, x_end = crop_image(sys.argv[1])
model = load('model/model.joblib')
# Divide the sheet into segments (each containing only one line)
segments = segment_image(img)
# Process each segment on its own
for segment in segments:
    # segment = segment[:,x_start:x_end]
    # Change the segment image into a binary image
    gray_img = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    bin_img_s = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 53, 4)
    cv2.imwrite("segment"+str(segnum)+".jpg", segment)
    # Remove horizontal lines from the binary image
    bin_img = remove_lines_s(segment, segnum)
    cv2.imwrite("bin_img_after"+str(segnum)+".jpg", bin_img*255)
    # Detect ellipses from the binary image with the horizontal lines removed
    ellipses = detect_ellipses(bin_img)
    cv2.imwrite("ellipses" + str(segnum) + ".jpg", ellipses * 255)
    # Label ellipses in the debugging picture in blue [optional]
    test_img = np.copy(segment)
    test_img[ellipses] = (255, 0, 0)
    # Detect centers of ellipses using center of mass
    centers = detect_centers(segment, ellipses)
    # Removal of ellipses from binary image, this can be used in debugging [optional]
    img_no_ellipses = remove_ellipses(ellipses, bin_img)*255
    cv2.imwrite("no_ellipses" + str(segnum) + ".jpg", img_no_ellipses)
    # Applying filters on the binary image to remove the remaining noise
    bin_img = cv2.medianBlur(bin_img.astype(np.uint8), 5)
    bin_img = cv2.medianBlur(bin_img.astype(np.uint8), 5)
    bin_img = cv2.medianBlur(bin_img.astype(np.uint8), 5)
    bin_img = bin_img.astype(np.uint8)
    # Invert the image so that the notes would be white and the background black (range: 0-255)
    bin_img = np.logical_not(bin_img)
    kernel = np.ones((3, 3), np.uint8)
    bin_img = (bin_img*255).astype(np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=2)
    # Sort the centers of the ellipses obtained on the direction of the x-axis
    centers = centers[centers[:, 1].argsort()]
    # Label each center as a + in red in the debugging picture [optional]
    for center in centers:
        # print(center)
        test_img[center[0], center[1]] = (0, 0, 255)
        test_img[center[0]-1, center[1]] = (0, 0, 255)
        test_img[center[0]+1, center[1]] = (0, 0, 255)
        test_img[center[0], center[1]-1] = (0, 0, 255)
        test_img[center[0], center[1]+1] = (0, 0, 255)
    # Label horizontal lines in green in the debugging picture [optional]
    linesImage = getLines(segment)
    llI = np.copy(linesImage) == 255
    test_img[llI] = (0, 255, 0)
    cv2.imwrite("debug" + str(segnum) + ".jpg", test_img)
    # Get the pitches of the note heads detected
    pitches = detectPositions(linesImage, centers, segnum == 0)
    # Segment the symbols using the vertical projection trick and bound them with rectangles
    br = vertical_segmentation(bin_img/255)
    pitches_array = draw_bounding(bin_img/255, segnum, br, pitches)
    # Print notes and add to song list
    print(pitches_array)
    print("#########################################")
    song.extend(pitches_array)
    # Increment segment number and go to the next symbol
    segnum += 1
# Change the song list of tuples into a tuple of tuples, make its corresponding wave file and play it
song = tuple(song)
print("SONG:", song)
pysynth.make_wav(song, fn="test.wav")
os.system("test.wav")
