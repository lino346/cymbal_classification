from cProfile import label
from enum import unique
from utils import Utils
from train import train_by_svm, svm
from test import test
import warnings
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pygrape import pygrape
Ziljian_DATASET_PATH = '/home/lino/Desktop/7100_s2/7100_s2_dataset/ziljian/'
Ziljian_LABEL_PATH = '/home/lino/Desktop/7100_s2/label/label-ziljian.txt'
Paiste_DATASET_PATH = '/home/lino/Desktop/7100_s2/7100_s2_dataset/paiste/'
Paiste_LABEL_PATH = '/home/lino/Desktop/7100_s2/label/label-paiste.txt'
SUBSET_DATASET_PATH = '/home/lino/Desktop/7100_s2/7100_s2_dataset/subset_for_runTest/'

def get_length(threshold_audio_len):
    ziljian_data_list = get_data_list_from_path(Ziljian_DATASET_PATH) 
    paiste_data_list = get_data_list_from_path(Paiste_DATASET_PATH)
    ziljian_data_filtered_list= []
    paiste_data_filtered_list = []
    # for audio_path in ziljian_data_list:
    #     # print(audio_path)
    #     x, sr = librosa.load(Ziljian_DATASET_PATH + audio_path + '.wav')
    #     # ziljian_data_len.append(round(len(x)/sr, 1))
    #     ziljian_data_len.append(int(len(x) / sr))
    #     # if int(len(x) / sr) < 1:
    #     #     mfcc = librosa.feature.mfcc(x[:22500], sr = sr, n_mfcc=13).T
    #     #     print(mfcc.shape)
    for idx, audio_path in enumerate(paiste_data_list):
        
        x, sr = librosa.load(Paiste_DATASET_PATH + audio_path + '.wav')
        audio_len = len(x) / sr
        count_filtered = np.zeros(5)
        if audio_len < threshold_audio_len:
            continue
        else:
            paiste_data_filtered_list.append(audio_path)
        
    # z_array = np.array(ziljian_data_len)
    # x_axis, y_axis = np.unique(z_array, return_counts=True)
    # p_array = np.array(paiste_data_len)
    # x_axis, y_axis = np.unique(p_array, return_counts=True)

    # plt.figure(figsize=(13,5))
    # plt.title('Paiste Audio Length Distribution')
    # plt.xlabel('Audio Length')
    # plt.ylabel('Numbers of Audio')
    # plt.hist(p_array, bins=x_axis.shape[0])
    # plt.show()

def length_distribution():
    ziljian_data_list = get_data_list_from_path(Ziljian_DATASET_PATH) 
    paiste_data_list = get_data_list_from_path(Paiste_DATASET_PATH)
    ziljian_data_label_list = get_data_label_from_list(ziljian_data_list, Ziljian_LABEL_PATH)
    paiste_data_label_list = get_data_label_from_list(paiste_data_list, Paiste_LABEL_PATH)
    length_0 = np.zeros(5)
    length_0_5= np.zeros(5)
    length_1 = np.zeros(5)
    length_2 = np.zeros(5)
    total = np.zeros(5)
    ziljian_data_filtered_list= []
    paiste_data_filtered_list = []
    # for idx, audio_path in enumerate(ziljian_data_list):
        
    #     x, sr = librosa.load(Ziljian_DATASET_PATH + audio_path + '.wav')
    #     audio_len = len(x) / sr
    #     # print(audio_len)
    #     if audio_len < 0.5:
    #         length_0[ziljian_data_label_list[idx]] += 1
    #     if (audio_len >= 0.5):
    #         length_0_5[ziljian_data_label_list[idx]] += 1
    #     if audio_len >= 1:
    #         length_1[ziljian_data_label_list[idx]] += 1
    #     if audio_len >= 2:
    #         length_2[ziljian_data_label_list[idx]] += 1
    #     total[ziljian_data_label_list[idx]] += 1
    # print(length_0 / total)
    # print(length_0_5 / total)
    # print(length_1 / total )
    # print(length_2 / total)
    # print(total / total)
    
    for idx, audio_path in enumerate(paiste_data_list):
        
        # x, sr = librosa.load(Paiste_DATASET_PATH + audio_path + '.wav')
        # audio_len = len(x) / sr
        # # print(audio_len)
        # if audio_len < 0.5:
        #     length_0[paiste_data_label_list[idx]] += 1
        # if (audio_len >= 0.5):
        #     length_0_5[paiste_data_label_list[idx]] += 1
        # if audio_len >= 1:
        #     length_1[paiste_data_label_list[idx]] += 1
        # if audio_len >= 2:
        #     length_2[paiste_data_label_list[idx]] += 1
        total[paiste_data_label_list[idx]] += 1
    print(length_0 )
    print(length_0_5 )
    print(length_1 )
    print(length_2 )
    print(total )

def filter_data_listBYlength(threshold_audio_len):
    ziljian_data_list = get_data_list_from_path(Ziljian_DATASET_PATH) 
    paiste_data_list = get_data_list_from_path(Paiste_DATASET_PATH)
    ziljian_data_label_list = get_data_label_from_list(ziljian_data_list, Ziljian_LABEL_PATH)
    paiste_data_label_list = get_data_label_from_list(paiste_data_list, Paiste_LABEL_PATH)
    ziljian_data_filtered_list= []
    paiste_data_filtered_list = []
    writer = pygrape(0.5)
    for idx, audio_path in enumerate(paiste_data_list):
        x, sr = librosa.load(Paiste_DATASET_PATH + audio_path + '.wav')
        audio_len = len(x) / sr
        writer.writer(" " + str(round(idx/len(paiste_data_list), 4)*100) + "%")
        writer.flush()
        if audio_len < threshold_audio_len:
            continue
        else:
            paiste_data_filtered_list.append(audio_path)
    writer.stop()
    writer = pygrape(0.4)
    for idx, audio_path in enumerate(ziljian_data_list):
        x, sr = librosa.load(Ziljian_DATASET_PATH + audio_path + '.wav')
        audio_len = len(x) / sr
        writer.writer(" " + str(round(idx/len(ziljian_data_list), 4)*100) + "%")
        writer.flush()
        if audio_len < threshold_audio_len:
            continue
        else:
            ziljian_data_filtered_list.append(audio_path)
    writer.stop()
    
    return ziljian_data_filtered_list, paiste_data_filtered_list
    
   
def feature_visual(z_data_path, z_label_path, p_data_path, p_label_path):
    z_data = get_data_list_from_path(z_data_path)
    p_data = get_data_list_from_path(p_data_path)
    z_label = get_data_label_from_list(z_data, z_label_path)
    p_label = get_data_label_from_list(p_data, p_label_path)

    label = np.array(p_label)
    # label = np.array(z_label+p_label)
    features_high, label, mean, std = features_concatenate(p_data, p_label, p_data_path, 'train')
    np.save('/home/lino/Desktop/7100_s2/features_paiste_same.npy',features_high)
    features_paiste_same = np.load('/home/lino/Desktop/7100_s2/features_paiste_same.npy')

    # features_ziljian = np.load('/home/lino/Desktop/7100_s2/features_ziljian_nonsame.npy')
    # features_paiste = np.load('/home/lino/Desktop/7100_s2/features_paiste_nonsame.npy')
    # feature_combine = np.vstack((features_ziljian, features_paiste))
    
    feature = TSNE(n_components=2).fit_transform(features_paiste_same)
    plt.scatter(feature[label == 0][:,0],feature[label == 0][:,1])
    plt.scatter(feature[label == 1][:,0],feature[label == 1][:,1])
    plt.scatter(feature[label == 2][:,0],feature[label == 2][:,1])
    plt.scatter(feature[label == 3][:,0],feature[label == 3][:,1])
    plt.scatter(feature[label == 4][:,0],feature[label == 4][:,1])
    plt.title('Feature Distribution in Paiste (same length)')
    plt.legend(['crash', 'ride', 'china', 'splash', 'hh'])
    plt.show()


def type_distribution():
    classification_task = 'type'

    utils = Utils(train='Ziljian', label=classification_task, process='whole')
    
    train_data_list, test_data_list = utils.get_data_list_from_npy(train='Ziljian')

    train_data_label_list, test_data_label_list = utils.get_data_label_from_list()
    
    
    unique_z, unique_z_num = np.unique(train_data_label_list, return_counts=True)
    unique_p, unique_p_num = np.unique(test_data_label_list, return_counts=True)
    # # a = np.array(train_data_label_list)
    # # b = np.where(a==6)
    # # print(b)
    print(unique_z)
    print(unique_z_num)
    print(unique_p)
    print(unique_p_num)
    # # print(train_data_list)
    # # print(train_data_label_list)
    # # print(test_data_list)
    # # print(test_data_label_list)
    # plt.subplot(2,1,1)
    # plt.bar(unique_z, unique_z_num, align='center')
    # plt.title('Ziljian Diameter Distribution')
    # plt.xlabel('Diameter')
    # plt.ylabel('Numbers')
    # plt.subplot(2,1,2)
    # plt.bar(unique_p, unique_p_num, align='center')
    # plt.title('Paiste Diameter Distribution')
    # plt.xlabel('Diameter')
    # plt.ylabel('Numbers')
    # # print(unique_z, unique_z_num)
    # # print(unique_p, unique_p_num)
    # plt.show()


def diameter_distribution():
    classification_task = 'diameter'

    utils = Utils(train='Ziljian', label=classification_task, process='whole')
    
    train_data_list, test_data_list = utils.get_data_list_from_npy(train='Ziljian')

    train_data_label_list, test_data_label_list = utils.get_data_label_from_list()
    train_data_label_array = np.array(train_data_label_list)
    test_data_label_array = np.array(test_data_label_list)
    
    unique_z, unique_z_num = np.unique(train_data_label_list, return_counts=True)
    unique_p, unique_p_num = np.unique(test_data_label_list, return_counts=True)

    print(unique_z)
    print(unique_z_num)
    print(unique_p)
    print(unique_p_num)
    # a = np.array(train_data_label_list)
    # b = np.where(a==6)
    # print(b)
    # print(unique_z)
    # print(unique_z_num)
    # print(train_data_list)
    # print(train_data_label_list)
    # print(test_data_list)
    # print(test_data_label_list)
    # plt.subplot(2,1,1)
    # plt.bar(unique_z, unique_z_num, align='center', color = 'r', label = 'z')
    # plt.title('Ziljian Diameter Distribution')
    # plt.xlabel('Diameter')
    # plt.ylabel('Numbers')
    # # plt.subplot(2,1,2)
    # plt.bar(unique_p, unique_p_num, align='center', color = 'b', label = 'p')
    # plt.title('Paiste Diameter Distribution')
    # plt.xlabel('Diameter')
    # plt.ylabel('Numbers')
    # plt.legend
    # print(unique_z, unique_z_num)
    # print(unique_p, unique_p_num)
    # plt.show()


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # length_distribution()
        # a, b = filter_data_listBYlength(0.5)
        # print(len(a), len(b))
        # type_distribution()
        diameter_distribution()
        # feature_visual(Ziljian_DATASET_PATH, Ziljian_LABEL_PATH, Paiste_DATASET_PATH, Paiste_LABEL_PATH)
        # print(int(1.9))