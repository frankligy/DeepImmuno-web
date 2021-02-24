'''
Accept peptide and MHC input,
compute immunogenicity


'''

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler,MinMaxScaler,RobustScaler
import collections
import re
from mhcflurry import Class1PresentationPredictor






def add_X(array):
    me = np.mean(array)
    array = np.append(array, me)
    return array


def read_index(path):
    with open(path, 'r') as f:
        data = f.readlines()
        array = []
        for line in data:
            line = line.lstrip(' ').rstrip('\n')
            line = re.sub(' +', ' ', line)

            items = line.split(' ')
            items = [float(i) for i in items]
            array.extend(items)
        array = np.array(array)
        array = add_X(array)
        Index = collections.namedtuple('Index',
                                       ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                        'T', 'W', 'Y', 'V', 'X'])
        I = Index._make(array)
    return I, array  # namedtuple


def read_all_indices():
    table = np.empty([21, 566])
    for i in range(566):
        if len(str(i)) == 1:
            ii = '00' + str(i)
        elif len(str(i)) == 2:
            ii = '0' + str(i)
        else:
            ii = str(i)

        NA_list_str = ['472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '520', '523', '524']
        NA_list_int = [int(i) for i in NA_list_str]
        if ii in NA_list_str: continue

        path = 'AAindex1/index{0}.txt'.format(ii)

        _, array = read_index(path)

        table[:, i] = array
    table = np.delete(table, NA_list_int, 1)
    return table


def scaling(table):  # scale the features
    table_scaled = RobustScaler().fit_transform(table)
    return table_scaled


def wrapper_read_scaling():
    table = read_all_indices()
    table_scaled = scaling(table)
    return table_scaled


def pca_get_components(result):
    pca= PCA()
    pca.fit(result)
    result = pca.explained_variance_ratio_
    sum_ = 0
    for index,var in enumerate(result):
        sum_ += var
        if sum_ > 0.95:
            return index    # 25 components



def pca_apply_reduction(result):   # if 95%, 12 PCs, if 99%, 17 PCs, if 90%,9 PCs
    pca = PCA(n_components=12)  # or strictly speaking ,should be 26, since python is 0-index
    new = pca.fit_transform(result)
    return new



'''
Here we sneak our CNN model in without changing too much of original pages
'''

def seperateCNN():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))

    x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)  # 9
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)     # 32
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
    y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2,outputs=y)

    combined = layers.concatenate([x.output,y.output])
    z = layers.Dense(128,activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1,activation='sigmoid')(z)

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model

def pull_peptide_aaindex(dataset):
    result = np.empty([len(dataset),10,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_aaindex(dataset):
    result = np.empty([len(dataset),46,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result



def pull_label_aaindex(dataset):
    result = np.empty([len(dataset),1])
    for i in range(len(dataset)):
        result[i,:] = dataset[i][2]
    return result

def aaindex(peptide,after_pca):

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)   # [12,21]
    encoded = np.empty([len(peptide), 12])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded


def peptide_data_aaindex(peptide,after_pca):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = aaindex(peptide,after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

def dict_inventory(inventory):
    dicA, dicB, dicC = {}, {}, {}
    dic = {'A': dicA, 'B': dicB, 'C': dicC}

    for hla in inventory:
        type_ = hla[4]  # A,B,C
        first2 = hla[6:8]  # 01
        last2 = hla[8:]  # 01
        try:
            dic[type_][first2].append(last2)
        except KeyError:
            dic[type_][first2] = []
            dic[type_][first2].append(last2)

    return dic


def rescue_unknown_hla(hla, dic_inventory):
    type_ = hla[4]
    first2 = hla[6:8]
    last2 = hla[8:]
    big_category = dic_inventory[type_]
    #print(hla)
    if not big_category.get(first2) == None:
        small_category = big_category.get(first2)
        distance = [abs(int(last2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
    else:
        small_category = list(big_category.keys())
        distance = [abs(int(first2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])






def hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory):    # return numpy array [34,12,1]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode,hla_type

def construct_aaindex(ori,hla_dic,after_pca,dic_inventory):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_aaindex(peptide,after_pca)    # [10,12]

        encode_hla,hla_type = hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory)   # [46,12]
        series.append((encode_pep, encode_hla, immuno))
    return series,hla_type

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic




def computing_s(peptide,mhc):
    # print(peptide)
    # print(mhc)
    # print(type(peptide))
    # print(type(mhc))
    table_scaled = wrapper_read_scaling()   # [21,553]
    after_pca = pca_apply_reduction(table_scaled)   # [21,12]
    hla = pd.read_csv('hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    '''
    Old version we use ResLike
    '''
    #ResLikeCNN_index = model_aaindex()
    #ResLikeCNN_index.load_weights('aaindex12_encoding_ReslikeCNN_reproduce/')

    '''
    New version we use CNN
    '''

    cnn_model = seperateCNN()
    cnn_model.load_weights('cnn_model_331_3_7/')

    peptide_score = [peptide]
    hla_score = [mhc]
    immuno_score = ['0']
    ori_score = pd.DataFrame({'peptide':peptide_score,'HLA':hla_score,'immunogenicity':immuno_score})
    dataset_score,hla = construct_aaindex(ori_score,hla_dic,after_pca,dic_inventory)
    input1_score = pull_peptide_aaindex(dataset_score)
    input2_score = pull_hla_aaindex(dataset_score)
    label_score = pull_label_aaindex(dataset_score)
    scoring = cnn_model.predict(x=[input1_score,input2_score])

    # label if it is a non-legit onoe
    non_legit = ['HLA-A*0203', 'HLA-A*0204', 'HLA-A*0207', 'HLA-A*0209', 'HLA-A*0210', 'HLA-A*0211', 'HLA-A*0213', 'HLA-A*0216', 'HLA-A*0217', 'HLA-A*0220', 'HLA-A*2403', 'HLA-A*2407', 'HLA-A*2602', 'HLA-A*2603', 'HLA-A*2901', 'HLA-A*3205', 'HLA-A*3301', 'HLA-A*3303', 'HLA-A*3402', 'HLA-A*6901', 'HLA-A*7401', 'HLA-A*8001', 'HLA-B*0706', 'HLA-B*0802', 'HLA-B*1301', 'HLA-B*1502', 'HLA-B*1503', 'HLA-B*1510', 'HLA-B*2701', 'HLA-B*2702', 'HLA-B*2704', 'HLA-B*2706', 'HLA-B*2709', 'HLA-B*3502', 'HLA-B*3508', 'HLA-B*3514', 'HLA-B*3701', 'HLA-B*3906', 'HLA-B*4006', 'HLA-B*4102', 'HLA-B*4201', 'HLA-B*4403', 'HLA-B*4405', 'HLA-B*4501', 'HLA-B*4801', 'HLA-B*5001', 'HLA-B*5201', 'HLA-B*5401', 'HLA-B*5601', 'HLA-B*5802', 'HLA-B*8101', 'HLA-C*0102', 'HLA-C*0303', 'HLA-C*0304', 'HLA-C*0401', 'HLA-C*0501', 'HLA-C*0602', 'HLA-C*0801', 'HLA-C*0802', 'HLA-C*1402', 'HLA-C*1502', 'HLA-C*1601']
    if hla in non_legit:
        flag = 'on'
    else:
        flag = 'off'
    return float(scoring),flag

def computing_m(peptide,mhc,is_checked):    # multiple MHC query
    table_scaled = wrapper_read_scaling()   # [21,553]
    after_pca = pca_apply_reduction(table_scaled)   # [21,12]
    hla = pd.read_csv('hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    cnn_model = seperateCNN()
    cnn_model.load_weights('cnn_model_331_3_7/')

    hla_score = ['HLA-A*0101', 'HLA-A*0201', 'HLA-A*0202', 'HLA-A*0301', 'HLA-A*1101', 'HLA-A*2402', 'HLA-A*6802', 'HLA-B*0702', 'HLA-B*0801', 'HLA-B*3501', 'HLA-B*4402']

    peptide_score = [peptide] * len(hla_score)
    immuno_score = ['0'] * len(hla_score)
    ori_score = pd.DataFrame({'peptide':peptide_score,'HLA':hla_score,'immunogenicity':immuno_score})
    dataset_score,hla_type = construct_aaindex(ori_score,hla_dic,after_pca,dic_inventory)
    input1_score = pull_peptide_aaindex(dataset_score)
    input2_score = pull_hla_aaindex(dataset_score)
    label_score = pull_label_aaindex(dataset_score)
    scoring = cnn_model.predict(x=[input1_score,input2_score])
    ori_score['immunogenicity'] = scoring
    ori_score.sort_values(by=['immunogenicity'],ascending=False,inplace=True)
    top5 = ori_score.iloc[0:5]
    
    p = top5['peptide'].tolist()
    m = top5['HLA'].tolist()
    i = [item for item in top5['immunogenicity']]

    # for these 5 complex, compute binding affnity
    if is_checked == 'True':

        '''
        strange input requirement: when you have 5 peptides, 5 mhc, you have to construct like this:
        a = predictor.predict(
            peptides=["NLVPMVATV","AAAAAAAAA","TTTTTTTT","PPPPPPPP","QQQQQQQQ"],
            alleles={'sample0': ['HLA-C*0517'], 'sample1': ['HLA-C*0602'], 'sample2': ['HLA-C*0401'], 'sample3': ['HLA-B*4403'], 'sample4': ['HLA-B*5101']},
            verbose=0)

        then since they compute a cross-product, I need to pick the value I need from the returned result
        '''

        tmp_dic_for_alleles= {}
        for index,mhc_ in enumerate(m):
            tmp_dic_for_alleles['sample{}'.format(index)] = [mhc_]
        predictor = Class1PresentationPredictor.load()
        result = predictor.predict(
            peptides=p,
            alleles=tmp_dic_for_alleles,
            verbose=0)
        final = []
        for sample,chunk in result.groupby(by='sample_name'):
            index = int(sample[-1:])
            final.append(chunk.iloc[index,:]['presentation_score'])
    else:
        final = ['NA','NA','NA','NA','NA']
    return p,m,i,final


def hla_convert(hla):
    hla = hla.replace('*','')
    return hla

def svg_path(hla):
    path = ["/static/{0}_positive_9.png".format(hla),"/static/{0}_negative_9.png".format(hla),"/static/{0}_positive_10.png".format(hla),"/static/{0}_negative_10.png".format(hla)]


def wrapper_file_process():
    cond = True
    try: file_process()
    except Exception as err: 
        #cond = False
        print(err)
    return cond

def file_process(upload="./uploaded/multiple_query.txt",download="./app/download/result.txt"):
    table_scaled = wrapper_read_scaling()   # [21,553]
    after_pca = pca_apply_reduction(table_scaled)   # [21,12]
    hla = pd.read_csv('hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    cnn_model = seperateCNN()
    cnn_model.load_weights('cnn_model_331_3_7/')
    
    ori_score = pd.read_csv(upload,sep=',',header=None)
    ori_score.columns = ['peptide','HLA']
    ori_score['immunogenicity'] = ['0'] * ori_score.shape[0]
    print('************************ 1 *************************')
    ori_score.to_csv(download,sep='\t',index=None)
    dataset_score,hla_type = construct_aaindex(ori_score,hla_dic,after_pca,dic_inventory)
    
    input1_score = pull_peptide_aaindex(dataset_score)
    input2_score = pull_hla_aaindex(dataset_score)
    label_score = pull_label_aaindex(dataset_score)
    scoring = cnn_model.predict(x=[input1_score,input2_score])
    scoring = cnn_model.predict(x=[input1_score,input2_score])
    ori_score['immunogenicity'] = scoring
    ori_score.to_csv(download,sep='\t',index=None)


def check_peptide(peptide):
    cond = True
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    if len(peptide) != 9 and len(peptide) != 10:
        cond = False
    elif not all(c in amino for c in peptide):
        cond = False
    return cond

def check_mhc(mhc):
    cond = True
    import re
    # HLA-A*0201
    z = re.match(r"^HLA-[ABC]\*\d{4}$",mhc)
    if not z:
        cond = False
    return cond


def binding_score_from_mhcflurry_s(peptide,mhc):
    predictor = Class1PresentationPredictor.load()
    result = predictor.predict(
        peptides=[peptide],
        alleles=[mhc],
        verbose=0)
    binding = result.iloc[0]['presentation_score']
    return float(binding)





    
      