import sys


import configparser


class PreProcessor:
    def __init__(self,config_file):
        config = configparser.ConfigParser()
   #     config_file = './data/config/config_properties.ini'
        config.read(config_file)
        self.kwargs = {}
        try:
            header = "DEPRESSION"
            self.kwargs['input'] = config.get(header, 'input')
            self.kwargs['output'] = config.get(header, "output")
            self.kwargs['util']  = config.get(header, "util")
            self.kwargs['liwc']  = config.get(header, "liwc")
            self.kwargs['features']  = config.get(header, "features")
            self.kwargs['feature_numbers']  = int(config.get(header, "feature_numbers"))
            self.kwargs['trainingFile']  = config.get(header, "trainingFile")
            
            
            
        except :
            print("check the parameters that you entered in the config file")
            exit()
        
        print (str(self.kwargs))
        
    def getArgs(self):
        return self.kwargs
        
    def set_target(self,target):
        self.target = target