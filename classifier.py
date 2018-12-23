from imutils import paths
from commonfunctions import *
import numpy as np
import argparse
import imutils
import cv2
import os
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from scipy.signal import find_peaks
path = 'eighthall'
imagePaths_eighth = list(paths.list_images(path))
path = 'quarterall'
imagePaths_quarter = list(paths.list_images(path))

def binarize(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img//255
    return img



def get_feature1(img):
    return img.shape[0]/img.shape[1]

def get_feature2(img):
    #plt.show()
    #bar(np.arange(img.shape[1]), np.sum(img,axis=0))
    peaks, _ = find_peaks(np.sum(img,axis=0),height=40,distance=7)
    return len(peaks)

def get_feature3(img):
    #plt.show()
    #bar(np.arange(img.shape[1]), np.sum(img,axis=0))
    return np.sum(img)


def get_feature4(img):
    sizeAfterResize = (25,25)
    image = cv2.resize(img, sizeAfterResize)
    return image.flatten()

def get_feature5(img):
    return cv2.HuMoments(cv2.moments(img)).flatten()
def get_all_features(img):
    features = []
    #features.append(get_feature1(img))
    #features.append(get_feature2(img))
    #features.append(get_feature3(img))
    features.extend(get_feature5(img))
    return features



def predict(clf,img):
    features = get_all_features(img)
    features = np.asarray(features)
    features = features.reshape((1, len(features)))
    return clf.predict(features)[0]

'''
training_data = []
labels = []
for (i, imagePath) in enumerate(imagePaths_eighth):
    img = cv2.imread(imagePath)
    img = binarize(img)
    features = get_all_features(img)
    features = np.asarray(features)
    training_data.append(features)
    #print(features)
    labels.append(0)

for (i, imagePath) in enumerate(imagePaths_quarter):
    img = cv2.imread(imagePath)
    img = binarize(img)
    features = get_all_features(img)
    features = np.asarray(features)
    training_data.append(features)
    print(features)
    labels.append(1)

training_data = np.asarray(training_data)
labels = np.asarray(labels)

img = cv2.imread("test/57.jpg")
img = binarize(img)
features_test = []
features_test = get_all_features(img)
features_test = np.asarray(features_test)
# print(features_test)
model = KNeighborsClassifier(n_neighbors=3)


model.fit(training_data, labels)
features_test = features_test.reshape((1, training_data.shape[1]))
print(model.predict(features_test))
dump(model, 'model.joblib')


'''
