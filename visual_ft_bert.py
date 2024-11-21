
import scipy
import argparse
import os,random
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import _reduction as _Reduction
from torch.nn import functional as F
from torch import Tensor
from scipy import ndimage
from torchvision import transforms
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
from transformers_multi_label_classification import BERTClass
import pandas as pd
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN=300

def inputseq(seq):
    comment_text = seq
    comment_text = " ".join(comment_text.split())

    inputs = tokenizer.encode_plus(
        comment_text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True
    )
    
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]


    return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
    }


        
if __name__ == '__main__':

    p=os.path.join('/Users','melodyu','Desktop','code.nosync','bert_classication','cleaned_labels','south_africa_2.csv')

    df = pd.read_csv(p)
    df['list'] = df[df.columns[6:]].values.tolist()
    new_df = df[['seq', 'list']].copy()
    print('dataset',len(new_df))
    # print('head',type(new_df))

    dir_to_save = os.path.join('/Users','melodyu','Desktop','code.nosync','bert_classication','model')
    filename_model = os.path.join(dir_to_save, 'bert_test_300.pth')  #model_decoder_nocut_90/model_de_whole_1024_145/model_apom15_de_256_100/model_apomND_de__qf_1024_100/model_simple_hiddim_50/model_Qfeature100_50/model_50/model_patchembed_nosinglelabel_nocut_30/model_patchembed_nosinglelabel_30/model_patchembed_30/model_newmat_prob__30/model_newmat_30/ model_orinewlabel30 / model_imgembed_16_30
    model=(torch.load(filename_model))
    model.eval()
  
    
    # ambigous / destructive / benefical / progrowth 1=yes ; 0=the opposite/other
    ############## test ##########################################################################
    
    seqs=[]
    pro=[]
    amb=[]
    ben=[]
    des=[]
    for i in range(689):
        seq=df['seq'][i]
        
        data=inputseq(seq)
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        
        # print(ids,mask,token_type_ids)
        
        outputs = model(ids.unsqueeze(0), mask.unsqueeze(0), token_type_ids.unsqueeze(0))
        # print('amb', outputs[0][0].item(),'des',outputs[0][1].item(),'ben',outputs[0][2].item(),'pro-growth',outputs[0][3].item())
        if outputs[0][3].item()<0:
            seqs.append(seq)
            pro.append(outputs[0][3].item())
            amb.append(outputs[0][0].item())
            ben.append(outputs[0][2].item())
            des.append(outputs[0][1].item())
    
    dataframe = pd.DataFrame({'seq':seqs,'pro-growth':pro,'env_amb':amb,'env_des':des,'env_ben':ben})
    dataframe.to_csv("not_pro.csv",index=False,sep=',')
    