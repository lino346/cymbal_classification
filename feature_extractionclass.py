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
import warnings
DATA_PATH = '/home/lino/Desktop/7100_s2/audio/'
LABEL_PATH = '/home/lino/Desktop/7100_s2/label/label.txt'
DATA_TEST_PATH = '/home/lino/Desktop/7100_s2/audio_test/'

class Feature (Utils):
    def __init__(self, train, label, process):
        super(Feature, self).__init__(train, label, process)

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


    def feature_mfcc(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        mfcc = librosa.feature.mfcc(x[:11250], sr = int(sr), n_mfcc = 13).T
        vector = np.hstack((mfcc.mean(0), mfcc.std(0)))

        return vector

    def feature_ad(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        vector = dur

        return vector

    def feature_centroid(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur= len(x) / sr

        centroid = librosa.feature.spectral_centroid(x[:11250], sr = int(sr), n_fft=512, hop_length=256).T
        vector = np.hstack((centroid.mean(0), centroid.std(0)))

        return vector    

    def feature_flat(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        flat = librosa.feature.spectral_flatness(x[:11250], n_fft=512, hop_length=256).T
        vector = np.hstack((flat.mean(0), flat.std(0)))
        
        return vector

    def feature_rolloff(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        rolloff = librosa.feature.spectral_rolloff(x[:11250], sr = int(sr), n_fft=512, hop_length=256).T
        vector= np.hstack((rolloff.mean(0), rolloff.std(0)))
        
        return vector

    def feature_bw(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        bw = librosa.feature.spectral_bandwidth(x[:11250], sr=int(sr), n_fft=512, hop_length=256).T
        vector = np.hstack((bw.mean(0), bw.std(0)))
        
        return vector

    def feature_contrast(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        contrast = librosa.feature.spectral_contrast(x[:11250], sr=int(sr), n_fft=512, hop_length=256).T
        vector = np.hstack((contrast.mean(0), contrast.std(0)))
        
        return vector

    def feature_rms(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        flat = librosa.feature.rms(x[:11250], frame_length=512, hop_length=256).T
        vector = np.hstack((flat.mean(0), flat.std(0)))
        
        return vector

    def feature_zcr(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr
        zcrs = librosa.feature.zero_crossing_rate(x[0:11250], 512, 256).T
        vector = np.hstack((zcrs.mean(0), zcrs.std(0)))
        
        return vector

    def feature_chromagram(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        chroma = librosa.feature.chroma_stft(x[:11250], sr=int(sr), n_fft=512, hop_length=256).T
        vector = np.hstack((chroma.mean(0), chroma.std(0)))
        
        return vector

    def feature_cqt(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        cqt = librosa.feature.chroma_cqt(x[:11250], sr=int(sr), hop_length=256).T
        vector = np.hstack((cqt.mean(0), cqt.std(0)))
        
        return vector

    def feature_cens(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        cens = librosa.feature.chroma_cens(x[:11250], sr=int(sr), hop_length=256).T
        vector = np.hstack((cens.mean(0), cens.std(0)))
        
        return vector

    def feature_melspec(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        mel = librosa.feature.melspectrogram(x[:11250], sr=int(sr), n_fft=512, hop_length=256).T
        vector = np.hstack((mel.mean(0), mel.std(0)))
        
        return vector

    def feature_poly_0(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        poly = librosa.feature.poly_features(x[:11250], sr=int(sr), n_fft=512, hop_length=256, order=0).T
        vector = np.hstack((poly.mean(0), poly.std(0)))
        
        return vector        

    def feature_poly_1(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        poly = librosa.feature.poly_features(x[:11250], sr=int(sr), n_fft=512, hop_length=256, order=1).T
        vector = np.hstack((poly.mean(0), poly.std(0)))
        
        return vector  

    def feature_poly_2(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        poly = librosa.feature.poly_features(x[:11250], sr=int(sr), n_fft=512, hop_length=256, order=2).T
        vector = np.hstack((poly.mean(0), poly.std(0)))
        
        return vector  

    def feature_tonnetz(self, audio_path = 0, x = 0, sr = 0, dur = 0):
        if audio_path != 0:
            x, sr = librosa.load(audio_path)
            dur = len(x) / sr

        tonal = librosa.feature.tonnetz(x[:11250], sr=int(sr)).T
        vector = np.hstack((tonal.mean(0), tonal.std(0)))
        
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

    def feature_flux(self, audio_path):
        x, sr = librosa.load(audio_path)
        audio_len = len(x) / sr
        xb, t = self.block_audio(x, 512, 256, sr)
        X = self.compute_spectrogram(xb)
        X = np.c_[X[:, 0], X]
        afDeltaX = np.diff(X, 1, axis=1)

        # flux
        flux = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
        # print(flux.shape)
        vector_temp = np.hstack((flux.mean(0), flux.std(0)))
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
        features_train = self.feature_crest(self.train_data_path + self.train_data_list[0] + '.wav')
        writer = pygrape(0.5)
        print("Extracting Features of Training data...")
        for index, data in enumerate(self.train_data_list):
            if index != 0:
                feature_one = self.feature_crest(self.train_data_path + data + '.wav')
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

        features_test = self.feature_crest_p(p_data[0], p_sr[0], p_len[0])
        writer = pygrape(0.5)
        print("Extracting Features of Testing data...")
        for index, data in enumerate(self.test_data_list):
            if index != 0:
                feature_one = self.feature_crest_p(p_data[index], p_sr[index], p_len[index])
                # print(feature_one.shape)
                features_test = np.vstack((features_test, feature_one))
                writer.writer(" " + str(round(index/len(self.test_data_list), 4)*100) + "%")
                writer.flush()
            if index == len(self.test_data_list) -1:
                writer.writer(" " + str(100) + "%")
                writer.flush()
        writer.stop()
        # print(features_train.shape, features_test.shape)
        return features_train, features_test

    def norm_features(self, features_train, features_test):
        features_train_scaled, mean, std = self.scale_feature(features_train)
        features_test_scaled = (features_test - mean) / std
        return features_train_scaled, features_test_scaled

    def features_extraction_4_sfs(self, feature_list):
        features_train = self.feature_selection(self.train_data_path + self.train_data_list[0] + '.wav', 0, 0 ,0, feature_list)
        writer = pygrape(0.5)
        print("Extracting Features of Training data...")
        for index, data in enumerate(self.train_data_list):
            if index != 0:
                feature_one = self.feature_selection(self.train_data_path + data + '.wav', 0, 0, 0, feature_list)
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

        features_test = self.feature_selection(0, p_data[0], p_sr[0], p_len[0], feature_list)
        # features_test = self.feature_selection(self.test_data_path + self.test_data_list[0] + '.wav', 0, 0 ,0, feature_list)
        writer = pygrape(0.5)
        print("Extracting Features of Testing data...")
        for index, data in enumerate(self.test_data_list):
            if index != 0:
                feature_one = self.feature_selection(0, p_data[index], p_sr[index], p_len[index], feature_list)
                # feature_one = self.feature_selection(self.test_data_path + data + '.wav', 0, 0 ,0, feature_list)
                # print(feature_one.shape)
                features_test = np.vstack((features_test, feature_one))
                writer.writer(" " + str(round(index/len(self.test_data_list), 4)*100) + "%")
                writer.flush()
            if index == len(self.test_data_list) -1:
                writer.writer(" " + str(100) + "%")
                writer.flush()
        writer.stop()
        # print(features_train.shape, features_test.shape)
        return features_train, features_test

    def feature_selection(self, audio_path = 0, x = 0, sr = 0, dur = 0, feature_list = 0):
        for idx, feature_name in enumerate(feature_list):
            
            if feature_name == 'MFCC':
                feature_vector_tmp = self.feature_mfcc(audio_path, x, sr, dur)

            elif feature_name == 'Audio_dur':
                feature_vector_tmp = self.feature_ad(audio_path, x, sr, dur)

            elif feature_name == 'Centroid' :
                feature_vector_tmp = self.feature_centroid(audio_path, x, sr, dur)
            
            elif feature_name == 'Flatness' :
                feature_vector_tmp = self.feature_flat(audio_path, x, sr, dur)

            elif feature_name == 'Rolloff' :
                feature_vector_tmp = self.feature_rolloff(audio_path, x, sr, dur)

            elif feature_name == 'Bandwidth' :
                feature_vector_tmp = self.feature_bw(audio_path, x, sr, dur)

            elif feature_name == 'Contrast' :
                feature_vector_tmp = self.feature_contrast(audio_path, x, sr, dur)

            elif feature_name == 'RMS' :
                feature_vector_tmp = self.feature_rms(audio_path, x, sr, dur)            

            elif feature_name == 'ZCR' :
                feature_vector_tmp = self.feature_zcr(audio_path, x, sr, dur)

            elif feature_name == 'Chromagram' :
                feature_vector_tmp = self.feature_chromagram(audio_path, x, sr, dur)

            elif feature_name == 'CQT' :
                feature_vector_tmp = self.feature_cqt(audio_path, x, sr, dur)

            elif feature_name == 'CENS' :
                feature_vector_tmp = self.feature_cens(audio_path, x, sr, dur)

            elif feature_name == 'MelSpectrogram' :
                feature_vector_tmp = self.feature_melspec(audio_path, x, sr, dur)

            elif feature_name == '0th poly' :
                feature_vector_tmp = self.feature_poly_0(audio_path, x, sr, dur)

            elif feature_name == '1st poly' :
                feature_vector_tmp = self.feature_poly_1(audio_path, x, sr, dur)

            elif feature_name == '2nd poly' :
                feature_vector_tmp = self.feature_poly_2(audio_path, x, sr, dur)

            elif feature_name == 'TonalCentroid' :
                feature_vector_tmp = self.feature_tonnetz(audio_path, x, sr, dur)

            if idx == 0:
                feature_vector = feature_vector_tmp
            else:
                feature_vector = np.hstack((feature_vector, feature_vector_tmp))

        return feature_vector

if __name__ == '__main__':
    # utils = Utils(train='Ziljian', process='whole')
    # train_data_list, test_data_list = utils.get_data_list_from_npy(train='Ziljian')
    # train_data_label_list, test_data_label_list = utils.get_data_label_from_list()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature = Feature(train='Ziljian', process='whole')
        path = '/home/lino/Desktop/7100_s2/7100_s2_dataset/paiste/0000.wav'
        path2 = '/home/lino/Desktop/7100_s2/7100_s2_dataset/paiste_5/0023.wav'
        p_data = np.load('p_data.npy')
        p_sr = np.load('p_sr.npy')
        p_len = np.load('p_len.npy')
        x , sr = librosa.load(path)
        print(x[:12500])
        print(sr)
        print(p_data[0])
        print(int(p_sr[0]))
        b = feature.feature_cqt(0, p_data[0], int(p_sr[0]), p_len[0])
        c = feature.feature_cqt(path2)
