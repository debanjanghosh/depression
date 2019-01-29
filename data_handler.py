

from collections import defaultdict
import codecs
from collections import Counter
import numpy as np

class DataHandler:
    def  __init__(self,kwargs):
        print('inside data handler')
        self.kwargs = kwargs
        
    def loadEmbedding(self,vocabs,vector_length):
    
        path = '/Users/dg513/work/eclipse-workspace/scratch-workspace/ScratchProject/data/wordnet/glove.6B/'
        path='/Users/mit-gablab/work/data_workspace/pretrained_embeddings/glove.6B/'
    
        
        word_vec_file = 'glove.6B.100d.txt'
        
       # glove_path = '/home/z003ndhf/work_debanjan/data/glove_vectors/'
       # glove_file = glove_path + 'glove_' + source + '.txt'
        model = {}
        k = -1
    
        with open(path+word_vec_file,'r',encoding='latin1') as f:
            for line in f:
              #  print line
                features = line.split() 
                token = features[0]
                if token not in vocabs:
                    continue
                vector = np.array(features[1:],dtype="float32")
                model[token] = vector
                k = len(vector)
            
        print ('word model is loaded...')
        print ('# of word dimensions is '  + str(k))
        return model


    def loadAllVocabs(self,inputPath):
        
        cutoff = 5
        file1 = 'depression_all_data.txt'
        
        f = open(inputPath+file1)
        allWords = []
        for line in f:
       
            elements = line.strip().split('\t')
            
            quote = elements[6].lower().strip()
            allText = quote 
            words = allText.lower().split()
      #      words = nltk.word_tokenize(allText.lower(), language='english')
            allWords.extend(words)
        
        f.close()
        cntx = Counter( [ w for w in allWords ] )
        lst = [ x for x, y in cntx.items() if y > cutoff ]
    
        return lst






        
    def loadInputData(self,type):
    
        inputPath = self.kwargs.get('input')
        trainingFile = self.kwargs.get('trainingFile')


        input_kwargs = {}
        quotes = []
        categories = []
        category_map = defaultdict(int)
        f = codecs.open(inputPath+trainingFile,'r','utf8')
    
        for line in f:
            elements = line.strip().split('\t')
            dataType = elements[0]
            if dataType.lower() != type:
                continue 
            participant = elements[5]
            if participant != 'Participant':
                continue
            
            category = float(elements[2])
            categories.append(category)
            category_map[category]+=1
            quote = elements[6]
            quotes.append(quote)
            
        f.close()
        input_kwargs['input'] = quotes
        input_kwargs['labels'] = categories
        
        return input_kwargs

