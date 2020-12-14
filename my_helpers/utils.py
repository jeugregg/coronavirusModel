# -*- coding: utf-8 -*-
''' 
Module for utility helper functions 
'''

import datetime
import re
import shutil


# save file before update
def clean_file(path_file_name, flag_copy=False):
    '''
    Clean file already traited : rename file with date
    '''
    try:
        d = datetime.datetime.now()
        str_date = '_' + d.strftime("%Y%m%d_%H_%M_%S")
       
        res_re = re.search('\.\w+$', path_file_name)
        
        path_file_name_saved = \
            path_file_name[0:res_re.start()] + str_date + res_re.group(0)
        if flag_copy:
            shutil.copy(path_file_name, path_file_name_saved)
        else:
            shutil.move(path_file_name, path_file_name_saved)

        print('File {} moved!'.format(path_file_name_saved))
    except:
        print('File {} does not exist!'.format(path_file_name))