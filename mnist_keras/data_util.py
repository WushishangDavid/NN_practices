"""
Ref:http://www.csuldw.com/2016/02/25/2016-02-25-machine-learning-MNIST-dataset/
"""

import numpy as np
import struct


class DataUtils(object):

    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def getImage(self):
        binfile = open(self._filename, 'rb')    # Open a ninary file
        buf = binfile.read()
        binfile.close()
        index=0
        numMagic, numImg, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index) # return a tuple
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImg):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j]>1:
                    imgVal[j]=1
            images.append(imgVal)
        return np.array(images)

    def getLabel(self):
        binfile = open(self._filename, 'rb')    # Open a ninary file
        buf = binfile.read()
        binfile.close()
        index=0
        magic, numItems = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes)
        labels = []
        for i in range(numItems):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte)
            labels.append(im[0])
        return np.array(labels)




