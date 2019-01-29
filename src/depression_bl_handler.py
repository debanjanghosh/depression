from _elementtree import Element
import codecs
from collections import Counter
import csv
import logging
import math
import operator
import os
from random import shuffle
import string
import sys

import nltk
from nltk.tbl import feature
from numpy import float16
from scipy import sparse
import scipy
from scipy.sparse.coo import coo_matrix
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm, metrics, naive_bayes
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from feature_generator import *
import numpy as np
from process_properties import *
from data_handler import *


#from template_functions import *
#from scipy import csr_matrix
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('./data/self_training.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

sentObject = SentimentIntensityAnalyzer()

def createRegularTrainingFeatureVector(featureList,input_kwargs,featureNames,featGenr,\
                                       feature_cache,maxFeatPosn,vectors,\
                                       unknown_vec,featPosnNorm):
    indptr = [0]    
    indices = []
    data = []

    labels =[]

    totalPos = 0
    totalNeg = 0 
    neutral = 0 
    
    training_data = input_kwargs['input'] 
    categories = input_kwargs['labels'] 
    
    for index,pair in enumerate(training_data):
        quote = pair
        label = categories[index]
        
        
        featureMap = {}
        
        ''' try each feature(s) set '''
                
        if 'TAGQ' in featureNames:
            tags = featGenr.generateTagqFeatures(quote,response,'TARGET')
            featureMap.update(tags)

        
        if 'INTERJ' in featureNames:
            interjs = featGenr.generateInterjFeatures(quote,response,'TARGET')
            featureMap.update(interjs)

        if 'EMBED' in featureNames:
            embeds = featGenr.generateEmbeddingFeatures(quote,vectors,unknown_vec,100)
            featureMap.update(embeds)
        
        if 'VADER' in featureNames:
            vader_senti = featGenr.getSentiVaderFeatures(quote, 'TARGET')
            featureMap.update(vader_senti)
            
        if 'JACCARD' in featureNames:
            jaccards = featGenr.generateJaccardFeatures(quote,response)
            featureMap.update(jaccards)

        if 'PUNCT' in featureNames:
            puncts = featGenr.generatePunctFeatures(response)
            featureMap.update(puncts)
        
        if 'MODAL' in featureNames:
            modals = featGenr.generateModalFeatures(quote,'TARGET')
            featureMap.update(modals)

        if 'DISCOURSE' in featureNames:
            discourses = featGenr.generateDiscourseFeatures(quote,'TARGET')
            featureMap.update(discourses)

        if 'AGREE_LEX' in featureNames:
            agreements = featGenr.generateAgreeDisagreeFeatures(quote,response,'TARGET')
            featureMap.update(agreements)
            
        if 'NGRAM' in featureNames:
            ngrams = featGenr.generateNGramFeatures(quote,'TARGET')
            featureMap.update(ngrams)

        if 'FIRSTLAST' in featureNames:
            ngrams = featGenr.generateLexFeatures(quote,'TARGET')
            featureMap.update(ngrams)

        if 'SENTI' in featureNames :
            sentis =  featGenr.generateMPQAFeatures(quote,'TARGET')
            featureMap.update(sentis)
        
        if 'HYPER' in featureNames:
            hypers = featGenr.generateHyperFeatures(quote,response,'TARGET')
            featureMap.update(hypers)
                
        if 'PUNCT' in featureNames:
            puncts = featGenr.generatePunctFeatures(response)
            featureMap.update(puncts)
            
        if 'EMOT' in featureNames:
            emots = featGenr.generateEmotFeatures(quote,response,'BOTH')
            featureMap.update(emots)
    
     #       liwcs = getLIWCFeatures(liwcFeats,argument)
     #       featureMap.update(liwcs)
        if 'HEDGE' in featureNames:
            hedge = featGenr.generateHedgeFeatures(quote,'TARGET')
            featureMap.update(hedge)
        
        if 'PERSON' in featureNames:
            persons = featGenr.getPersonFeatures(quote,'TARGET')    
            featureMap.update(persons)
            
        if 'SARC' in featureNames:
            sarc_map ={}
            sarc_map[sarcasm] = 1.0
            featureMap.update(sarc_map)
    
        if 'TOPVERBS' in featureNames:
            verbs = getVerbFeatures(kwargs['topverbs'],argument)
            featureMap.update(verbs)
            
        if 'LIWC' in featureNames:
            allLiwcFeatures = featGenr.generateLIWCFeatures(quote,'TARGET')
            featureMap.update(allLiwcFeatures)

            
     
      #      jaccard_bin = convertToBin(jaccard_arg,'jaccard_arg')
      #      if jaccard_bin is not None:
       #         featureMap[jaccard_bin] = 1.0

 #       fv = createFeatureVector(featureMap,featureList,maxFeatPosn,featPosnNorm)
        fv = createFeatureVectorForSVM(featureMap,featureList,maxFeatPosn,featPosnNorm)
        
        
     #   label = convert(label)
        if label == 1.0:
            totalPos+=1
        if label == 0.0:
            totalNeg+=1
            
        for feature in fv:
            indices.append(feature[0])
            data.append(feature[1])
        
        indptr.append(len(indices))

        labels.append(label)
    
    msg = 'total (pos,neg) is ' + str(totalPos) + ' ,' + str(totalNeg)
    logger.info(msg)
    
    trainingData = data,indices,indptr,labels

    return trainingData,feature_cache

def createFeatureVector(featureMap,allFeatures,maxFeatPosn,featPosnNorm):
    
    valList = []
    for index,key in enumerate(allFeatures):
        value = featureMap.get(key)
        if value is not None and value > 0 :
            valList.append(featPosnNorm+index) # = value
                    
    ''' there is an issue in # of features for train/test in scikit learn'''
    ''' we need to set explicitly the "max" of feature for train otherwise there is '''
    ''' len(Feature) issue'''  
    ''' SOLVED (above): use transform function?'''
    
#    if featureType == 'pattern':
#        return valList

    valList.append(maxFeatPosn)

    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        #valMap[len(allFeatures)] = 0.0
        #valList.append(len(allFeatures)-1) # = value
        valList.append(maxFeatPosn)
    '''
    return valList


'''
def getDiscourseMarkers(discourses,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower(), language='english')
    count = 0.0
    for word in words1:
        if 'discourse|||'+word.strip() in discourses:
            featureMap[ 'discourse|||'+word.strip()] =1.0
    
# not for the single words!!!
    for discourse in discourses:
        ds  = discourse.split()
        if len(ds)>1:
            discourse = discourse.replace('discourse|||','')
            disc_txt = discourse.replace(', ',' ').strip()
            text_nopunct = argument.translate(string.maketrans("",""), string.punctuation)
            if disc_txt.strip() in text_nopunct:
                featureMap[ 'discourse|||'+disc_txt.strip()] =1.0
 
    
    
    return featureMap
'''
'''
def getVerbFeatures(verbs,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower(), language='english')
    count = 0.0
    for word in words1:
        if 'verb|||'+word.strip() in verbs:
            featureMap[ 'verb|||'+word.strip()] =1.0
            
    return featureMap

def getNgrams(ngrams,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower().decode('utf8'), language='english')
    count = 0.0
    for word in words1:
        if 'unigram|||'+word.strip() in ngrams:
            featureMap[ 'unigram|||'+word.strip()] =1.0
            
    return featureMap




def getHedgeFeatures(hedges,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower(), language='english')
    count = 0.0
    for word in words1:
        if 'hedge|||'+word.strip() in hedges:
            featureMap[ 'hedge|||'+word.strip()] =1.0
    
    # not for the single words!!!
    for hedge in hedges:
        ds  = hedge.split()
        if len(ds)>1:
            hedge = hedge.replace('hedge|||','')
            hedge = hedge.replace(', ',' ').strip()
            text_nopunct = argument.translate(string.maketrans("",""), string.punctuation)
            if hedge.strip() in text_nopunct:
                featureMap[ 'hedge|||'+hedge.strip()] =1.0
 
    
    
    return featureMap

'''


def createFeatureVector(featureMap,allFeatures,maxFeatPosn,featPosnNorm):
    
    valList = []
    for index,key in enumerate(allFeatures):
        value = featureMap.get(key)
        if value is not None and value > 0 :
           # if 'JACCARD' in allFeatures:
           #     valList.append(value) # = value

            valList.append(featPosnNorm+index) # = value
                    
    ''' there is an issue in # of features for train/test in scikit learn'''
    ''' we need to set explicitly the "max" of feature for train otherwise there is '''
    ''' len(Feature) issue'''  
    ''' SOLVED (above): use transform function?'''
    
#    if featureType == 'pattern':
#        return valList

    valList.append(maxFeatPosn)

    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        #valMap[len(allFeatures)] = 0.0
        #valList.append(len(allFeatures)-1) # = value
        valList.append(maxFeatPosn)
    '''
    return valList


def createFeatureVectorForSVM(featureMap,allFeatures,maxFeatPosn,featPosnNorm):
    
    valList = []
    for index,key in enumerate(allFeatures):
        value = featureMap.get(key)

        if value is not None and value > 0  :

          #  svmBuffer = svmBuffer + str(index+1) + ':' + str(value) + ' '
            tuple = featPosnNorm+index,value
            valList.append(tuple)        
      #  svmBuffer = svmBuffer # + ' ' + fileIdStr
    
    ''' there is an issue in # of features for train/test in scikit learn'''
    ''' we need to set explicitly the "max" of feature for train otherwise there is '''
    ''' len(Feature) issue'''  
    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        svmBuffer = svmBuffer + str(len(allFeatures)-1+1) + ':' + '0.0' + ' '
    
    svmBuffer = svmBuffer.strip()
    return svmBuffer
    '''
    tuple = maxFeatPosn,0
    valList.append(tuple)
    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        #valMap[len(allFeatures)] = 0.0
        #valList.append(len(allFeatures)-1) # = value
        valList.append(maxFeatPosn)
    '''
    return valList


def avg_feature_vector(words, model, num_features, index2word_set):
        #function to average all words vectors in a given paragraph
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0

        #list containing names of words in the vocabulary
        #index2word_set = set(model.index2word) this is moved as input param for performance reasons
        for word in words:
            if word in index2word_set:
                nwords = nwords+1
                featureVec = np.add(featureVec, model[word])

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec
	
	  

def TrainAndValidate(X_train, Y_train,X_test,Y_test,Cs,norms,kernel):
    
    best = 0
    best_params = {'C': None, 'penalty': None}
    best_f1 = 0.0 
    best_c = 'BLANK'
    best_norm = 'BLANK'
    best_kernel = 'BLANK'
    for C in Cs:
        for norm in norms:
      #  lr = LogisticRegression(C=C, penalty=norm, class_weight='auto')
            lr1  = SVC(C=C,kernel=kernel,class_weight='balanced',probability=True)
            lr1.fit(X_train, Y_train)
       #     X_test_csr = X_test.tocsr()
            predictions_all = lr1.predict(X_test)
            acc_score_all = accuracy_score(Y_test, predictions_all)
            f1_score_all = f1_score(Y_test, predictions_all,average='macro')
            if f1_score_all > best_f1:
                best_f1 = f1_score_all
                best_c = C
                best_norm = norm
                best_kernel = kernel
            print ('cost and norm ' + str(C) + ' ' + str(norm))
            logger.info("Classification report for classifier %s:\n%s\n" % \
            (lr1, metrics.classification_report( Y_test, predictions_all,digits=3)))
    
    print ('best params (macro-avg): ',best_f1, best_c, best_norm, best_kernel)
    
        


def generateFeatureAndClassification(kwargs):

    inputPath = kwargs.get('input')
    svmPath =  kwargs.get('output')
    utilPath = kwargs.get('util')
    trainingFile = kwargs.get('trainingFile')
    feature_numbers =  int(kwargs.get('feature_numbers'))
    featureNames = kwargs.get('features')

    dataHandler = DataHandler(kwargs)

    featGenr = FeatureGenerator(featureNames,kwargs)
    logger.info('Feature names are initialized')
    
    
    if 'EMBED' in featureNames:
        vocabs = dataHandler.loadAllVocabs(inputPath)
        vectors = dataHandler.loadEmbedding(vocabs,vector_length=100)
 

    allFeatureList = featGenr.initFeatures()

    logger.info('features are initialized...')
 #   maxFeatPosn = len(allFeatureList)

    training_data = dataHandler.loadInputData(type='train')
    logger.info('training data loaded...')
    
    dev_data = dataHandler.loadInputData(type='dev')
    logger.info(trainingFile)
    
#    test_data = loadInputData(inputPath+trainingFile,type='test')
#    logger.info('test data loaded...')

    training_regular_cache = []
    dev_regular_cache = []
    test_regular_cache = []
    
    boundary_index = len(allFeatureList)#.index('BOUNDARY_FEATURE')
    vec_length = 100
    unknown_vec = np.random.normal(0,0.17,vec_length) 
    
    training_reg_features,training_regular_cache = createRegularTrainingFeatureVector \
    (allFeatureList,training_data,featureNames,featGenr,training_regular_cache,boundary_index,vectors,unknown_vec,0)
        
    regTrainingData, indices, indptr, Y_train = training_reg_features[0],training_reg_features[1],\
    training_reg_features[2],training_reg_features[3]
    X_train = scipy.sparse.csr_matrix((regTrainingData, indices, indptr))

  #  printFeatures(X_pat_train)
    dev_reg_features,dev_regular_cache = createRegularTrainingFeatureVector(allFeatureList,\
                                            dev_data,featureNames,featGenr,dev_regular_cache,boundary_index,vectors,unknown_vec,0)
        
    regDevData, indices, indptr,Y_dev = dev_reg_features[0],dev_reg_features[1],\
    dev_reg_features[2],dev_reg_features[3]
    X_dev = scipy.sparse.csr_matrix((regDevData, indices, indptr))

    logger.info('All vector is loaded...')
    
    allFeatureList.append('BLANK_FEATURE')

    feat_len = min(feature_numbers,len(allFeatureList))
    X_kbest_reg = SelectKBest(chi2, k=feat_len)
    X_best_train = X_kbest_reg.fit_transform(X_train, Y_train)
    
 #   regularFeatureList_chi2 = []
    print ('best features: ')
    for i in  X_kbest_reg.get_support(indices=True):
        print ('index ',i, allFeatureList[i].encode('utf-8'))
 #       regularFeatureList_chi2.append(allFeatureList[i])
    
#    feature_posns = [ i for i in X_kbest_reg.get_support(indices=True)]
    #for name in regularFeatureList_chi2:
     #   print name 
    X_best_dev = X_kbest_reg.transform(X_dev)
   # X_best_test = X_kbest_reg.transform(X_test)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax_train = min_max_scaler.fit_transform(X_best_train.toarray())
    X_minmax_dev = min_max_scaler.fit_transform(X_best_dev.toarray())

    print ('\n')
      
    Cs = [1, 10, 100,1000]
    norms = ['l2']
    
    ''' we have already selected the best cost (10), which is 10'''
    TrainAndValidate(X_minmax_train, Y_train,X_minmax_dev,Y_dev,Cs,norms,'linear')
 #   TrainAndValidate(X_minmax_train, Y_train,X_minmax_test,Y_dev,Cs,norms,'linear')

    #TrainAndPredict(X_best_train,Y_train,X_best_test,Y_test,best_C,best_norm,'linear')                

    TrainAndValidate(X_minmax_train, Y_train,X_minmax_dev,Y_dev,Cs,norms,'rbf')
    #TrainAndPredict(X_best_train,Y_train,X_best_test,Y_test,best_C,best_norm,'rbf')                

def loadParameters(configFile):
    
    processorObj = PreProcessor(configFile)
    return processorObj.getArgs()

def main(argv):
    
    kwargs = loadParameters(argv[1])
  #  input = kwargs['input']
  #  svm_op = kwargs['output']
  #  utilpath = kwargs['util']
 #   featureList = kwargs['features']
    
  #  featureList = ['NGRAM','HEDGE','LIWC','VADER', 'SENTI','MODAL','EMBED','PERSON',
   #                'DISCOURSE']
    
    
#    featureList = arguFeatList

   # featureList = ['NGRAM', 'MODAL','SENTI','AGREE_LEX',  \
    #               'DISCOURSE', 'VADER','EMBED','TAGQ', 'HYPER', 'LIWC', 'PUNCT', 'EMOT' ]
    
  #  featureList = ['PUNCT', 'INTERJ','LIWC','EMBED','EMOT','HYPER','NGRAM']
    
 #   featureList = ['EMBED']
    
    
    vectors =None
  #  featureList = ['TAGQ']

    generateFeatureAndClassification(kwargs)

if __name__ == '__main__':
	main(sys.argv[1:])
