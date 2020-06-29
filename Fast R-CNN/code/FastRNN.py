import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, losses
import numpy as np

class FastRCNN():

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def run(self, input, label):
        featureMap = self.convLayer(input, 100)
        
        roi = self.selectiveSearch(featureMap)
        featureVector = self.roiPooling(roi)
        softMaxdata = self.convLayer(featureVector, 100)
        BBdata = self.convLayer(featureVector, 100)
        p_u = self.softmax(softMaxdata)
        t = self.softmax(BBdata)

        return self.lossLayer(p_u, t, label.at)

    def selectiveSearch(self, input):
        roi = []

        return roi

    def roiPooling(self, roi): 
        featureVectorList = []

        return featureVectorList

    def convLayer(self, input, x):
        # conv and 최적화
        featureMap = tf.conv2d(input (x, x))

        return featureMap

    def softmax(self, input):
        p_u = []
        return p_u

    def lossLayer(self, p_u, t, at):
        sum = 0
        for i in range(len(t)):
            sum += self.smooth_L1(t[i] -at[i])

        return (-np.log(p_u) + sum )
    
    def smooth_L1(self, i):
        if np.abs(i) < 1:
            return 0.5*i*i
        else :
            return np.abs(i) - 0.5



    def boundingBoxRegression(self, input):
        tx = 0; ty = 0; tw = 0; th = 0;
        t = [tx, ty, tw, th]

        return t