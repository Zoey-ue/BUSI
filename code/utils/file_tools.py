# -*- coding: utf-8 -*-


import os
def check_exit_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)