#!/usr/bin/env python
# coding: utf-8

# In[2]:
import sys
sys.path.append('.')
from gpml.visualisation.graphviz import *
import pandas as pd



#df = pd.read_csv('data/ugr16/ugr_sample_100k.csv',skiprows = lambda x : x%100 != 0)

#df['Label'] = df['Label'].replace(['background','blacklist'],0)

#df['Label'] = df['Label'].replace(['anomaly-spam','dos','scan44','scan11','nerisbotnet'],1)

#print_graph(df,'ip', 'Label', 'Source IP', 'Destination IP',url = './graphrepresentation',title='ugr16_1k',attack_name='Scan, Dos and Spam' ,src_port='Source Port', dst_port='Destination Port', src_mac=0, dst_mac=0)

# In[ ]:

df = pd.read_csv('data/unsw_nb15/Unsw_example700k.csv',skiprows = lambda x : x%100 != 0)

print_graph(df,'ip', '48', 'Source IP', 'Destination IP',url = './graph_representation',title='Unsw_7k',attack_name='Diverse attacks' ,sport='Source Port', dport='Destination Port', src_mac=0, dst_mac=0)

#df = pd.read_csv('data/wustl_ehmls/wustl-ehmls-2020_excerpt_long.csv')
#df['Label'] = df['Label'].replace(['Background',' '],0)
#df['Label'] = df['Label'].replace(['Fuzzers','Analysis','Backdoor','DoS','Shellcode','Exploits','Generic','Reconnaissance','Worms'],1)
#print_graph(df,'ip', 'Label', 'SrcAddr', 'DstAddr',url = './graphrepresentation',title='wustl_ehmls',attack_name='spoofing' ,src_port='Sport', dst_port='Dport', src_mac=0, dst_mac=0)
