import librosa
import os
from matplotlib.pyplot import axis
import numpy as np
import sklearn
from pygrape import pygrape
import time
from sklearn import preprocessing
from audiolazy import *
from utils import Utils
import math
import scipy as sp
DATA_PATH = '/home/lino/Desktop/7100_s2/audio/'
LABEL_PATH = '/home/lino/Desktop/7100_s2/label/label.txt'
DATA_TEST_PATH = '/home/lino/Desktop/7100_s2/audio_test/'

class Feature (Utils):
    def __init__(self, train, process):
        super(Feature, self).__init__(train, process)

    def block_audio(self, x,blockSize,hopSize,fs):
        # allocate memory
        numBlocks = math.ceil(x.size / hopSize)
        xb = np.zeros([numBlocks, blockSize])
        # compute time stamps
        t = (np.arange(0, numBlocks) * hopSize) / fs

        x = np.concatenate((x, np.zeros(blockSize)),axis=0)

        for n in range(0, numBlocks):
            i_start = n * hopSize
            i_stop = np.min([x.size - 1, i_start + blockSize - 1])

            xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

        return (xb,t)

    def compute_hann(self, iWindowLength):
        return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

    def compute_spectrogram(self, xb):
        numBlocks = xb.shape[0]
        afWindow = self.compute_hann(xb.shape[1])
        X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
        
        for n in range(0, numBlocks):
            # apply window
            tmp = abs(sp.fft(xb[n,:] * afWindow))*2/xb.shape[1]
        
            # compute magnitude spectrum
            X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
            X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) #let's be pedantic about normalization

        return X
    
    def scale_feature(self, feature):
        scaler = preprocessing.StandardScaler().fit(feature)
        mfcc_scaled = scaler.transform(feature)
        return mfcc_scaled, scaler.mean_, scaler.scale_

    def feature_mfcc_p(self, x, sr, audio_len):
        # x, sr = librosa.load(audio_path)
        # audio_len = len(x) / sr
        mfcc = librosa.feature.mfcc(x[:11250], sr = sr, n_mfcc = 13).T
        vector_temp = np.hstack((mfcc.mean(0), mfcc.std(0)))
        vector = np.hstack((vector_temp, audio_len))

        return vector


    def feature_mfcc(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        mfcc = librosa.feature.mfcc(x[:11250], sr = sr, n_mfcc = 13).T
        vector_temp = np.hstack((mfcc.mean(0), mfcc.std(0)))
        vector = np.hstack((vector_temp, audio_len))

        return vector

    def feature_lsf(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        a = lpc.autocor(x[0:11250],10)
        l = lsf(a)
        l = np.array(l)
        vector = np.hstack((l.mean(0), l.std(0)))
        # vector = np.hstack((vector_temp, audio_len))
        return vector

    def feature_centroid(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        centroid = librosa.feature.spectral_centroid(x[:11250], sr = sr, n_fft=512, hop_length=256).T
        vector_temp = np.hstack((centroid.mean(0), centroid.std(0)))
        vector = np.hstack((vector_temp, audio_len))
        return vector

    def feature_flux(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        xb, t = self.block_audio(x, 512, 256, sr)
        X = self.compute_spectrogram(xb)
        
        X = np.c_[X[:, 0], X]
        # X = np.concatenate(X[:,0],X, axis=1)
        afDeltaX = np.diff(X, 1, axis=1)

        # flux
        flux = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
        # mfcc = librosa.feature.spectral_centroid(x[:11250], sr = sr, n_fft=512, hop_length=256).T
        vector_temp = np.hstack((flux.mean(0), flux.std(0)))
        vector = np.hstack((vector_temp, audio_len))
        return vector

    def feature_flat(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        flat = librosa.feature.spectral_flatness(x[:11250], n_fft=512, hop_length=256).T
        vector_temp = np.hstack((flat.mean(0), flat.std(0)))
        vector = np.hstack((vector_temp, audio_len))
        return vector

    def feature_crest(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        xb, t = self.block_audio(x, 512, 256, sr)
        X = self.compute_spectrogram(xb)
        
        crest = X.sum(axis = 1, keepdims = True)
        crest[crest == 0] =1
        crest = X.max(1) / crest
        # print(vtsc.shape)
        # mfcc = librosa.feature.spectral_centroid(x[:11250], sr = sr, n_fft=512, hop_length=256).T
        vector_temp = np.hstack((crest.mean(0), crest.std(0)))
        vector = np.hstack((vector_temp, audio_len))
        return vector

    def feature_rolloff(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        rolloff = librosa.feature.spectral_rolloff(x[:11250], sr = sr, n_fft=512, hop_length=256).T
        vector_temp = np.hstack((rolloff.mean(0), rolloff.std(0)))
        vector = np.hstack((vector_temp, audio_len))
        return vector
    
    def feature_zcr(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        zcrs = librosa.feature.zero_crossing_rate(x[0:11250], 512, 256)
        vector = np.hstack((zcrs.mean(0), zcrs.std(0)))
        # vector = np.hstack((vector_temp, audio_len))
        return vector

    def same_shape_label(self, feature_shape, label):
        label_array = np.zeros((feature_shape,1))
        label_array[label_array == 0] = label
        # print (label_array, file_name_array)
        return label_array

    def features_concatenate(self, train_data_list, data_label, data_path, process, scaler_mean=0, scaler_std=0):
        features = self.feature_mfcc(data_path + train_data_list[0] + '.wav')
        writer = pygrape(0.5)
        print("Extracting Features...")
        for index, data in enumerate(train_data_list):
            if index != 0:
                feature_one = self.feature_mfcc(data_path + data + '.wav')
                # print(feature_one.shape)
                features = np.vstack((features, feature_one))
                writer.writer(" " + str(round(index/len(train_data_list), 4)*100) + "%")
                writer.flush()
            if index == len(train_data_list) -1:
                writer.writer(" " + str(100) + "%")
                writer.flush()
        writer.stop()
        if process == 'train':
            features_scaled, mean, std = self.scale_feature(features)
            data_label = np.array(data_label) 
            return features_scaled, data_label, mean, std
        elif process == 'test': 
            features_scaled = (features - scaler_mean) / scaler_std
            data_label = np.array(data_label) 
            return features_scaled, data_label

    def features_extraction(self):
        features_train = self.feature_mfcc(self.train_data_path + self.train_data_list[0] + '.wav')
        writer = pygrape(0.5)
        print("Extracting Features of Training data...")
        for index, data in enumerate(self.train_data_list):
            if index != 0:
                feature_one = self.feature_mfcc(self.train_data_path + data + '.wav')
                # print(feature_one.shape)
                features_train = np.vstack((features_train, feature_one))
                writer.writer(" " + str(round(index/len(self.train_data_list), 4)*100) + "%")
                writer.flush()
            if index == len(self.train_data_list) -1:
                writer.writer(" " + str(100) + "%")
                writer.flush()
        writer.stop()
        
        p_data = np.load('p_data.npy')
        p_sr = np.load('p_sr.npy')
        p_len = np.load('p_len.npy')

        features_test = self.feature_mfcc_p(p_data[0], p_sr[0], p_len[0])
        writer = pygrape(0.5)
        print("Extracting Features of Testing data...")
        print(len(self.test_data_list))
        for index, data in enumerate(self.test_data_list):
            if index != 0:
                feature_one = self.feature_mfcc_p(p_data[index], p_sr[index], p_len[index])
                
                # print(feature_one.shape)
                features_test = np.vstack((features_test, feature_one))
                writer.writer(" " + str(round(index/len(self.test_data_list), 4)*100) + "%")
                writer.flush()
            if index == len(self.test_data_list) -1:
                writer.writer(" " + str(100) + "%")
                writer.flush()
        writer.stop()
        print(features_train.shape, features_test.shape)
        return features_train, features_test

    def norm_features(self, features_train, features_test):
        features_train_scaled, mean, std = self.scale_feature(features_train)
        features_test_scaled = (features_test - mean) / std
        return features_train_scaled, features_test_scaled
