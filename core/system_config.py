# -*- coding:utf-8 -*-

"""
@date: 2022/8/9 上午10:06
@summary:
"""
import os
import configparser

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(project_dir)
config_parser = configparser.ConfigParser()
config_file = os.path.join(project_dir, 'etc/system.conf')
config_parser.read(config_file)