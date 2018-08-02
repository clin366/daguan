# -*- coding: utf-8 -*-
"""
Created on Wed Jun 4 14:40:54 2018

@author: Eric
"""
import logging
import time
import os

class Logger(object):
    """This is used for recording"""
    def __init__(self, file_path, clevel=logging.INFO, Flevel=logging.DEBUG):
        self.file_path = file_path
        self.log_path = self.file_path + os.sep + "log"
        cur_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        self.log_file = self.log_path + os.sep + cur_date + ".log"
        self.logger = logging.getLogger(self.log_file)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        
        # Set cmd
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        
        # Set file
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
    
    def debug(self, message):
        self.logger.debug(message)
 
    def info(self,message):
        self.logger.info(message)
 
    def warn(self,message):
        self.logger.warn(message)
 
    def error(self,message):
        self.logger.error(message)
 
    def critical(self,message):
        self.logger.critical(message)

log = Logger("/home/chenyu/daguan")

