
import asyncio
import logging
import os
import json
import time
import re
from datetime import datetime
import argparse
from tqdm import tqdm
import random
from openai import AsyncOpenAI,OpenAI
from tqdm.asyncio import tqdm_asyncio
from zeno_build.models import lm_config



def rocks_MCQ():
    MAPPING={0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",}
    IMG_URL_PREFIX="https://huggingface.co/datasets/hexuan21/geo_proj/resolve/main/rocks"
    PROMPT="图片中的岩石是什么种类？请从给出的选项中选择正确的。"
    NUM_OPTIONS=5
    
    src_file="./data/rocks/anno.json"
    src_data=json.load(open(src_file,"r",encoding='utf-8'))
    
    all_options=list(set([item["anno"] for item in src_data]))
    
    dest_file="./data/rocks_MCQ.json"
    os.makedirs(os.path.dirname(dest_file),exist_ok=True)
    dest_data=[]
    
    for item in tqdm(src_data):
        idx=item["idx"]
        ref_ans=item["anno"]
        mis_options=random.sample([x for x in all_options if x !=ref_ans],NUM_OPTIONS-1)
        options=mis_options+[ref_ans]
        random.shuffle(options)
        correct_index=options.index(ref_ans)
        ans=MAPPING[correct_index]
        
        new_item={}
        new_item["编号"]=idx  
        new_item["图片链接"]=f"{IMG_URL_PREFIX}/{idx}.png"
        new_item["问题"]=PROMPT+"\n"
        for i in range(NUM_OPTIONS):
            new_item["问题"]+=f"{MAPPING[i]}. {options[i]}\n"
        new_item["答案"]=ans
        
        dest_data.append(new_item)
    
    with open(dest_file.replace(".json",".txt"), 'w',encoding='utf-8') as file:
        for d in dest_data:
            for key, value in d.items():
                file.write(f"{key}: {value}")
                if key in ["编号","图片链接","答案"]:
                    file.write("\n")
            file.write("\n")
    
    with open(dest_file,"w",encoding='utf-8') as f:
        json.dump(dest_data,f,indent=4)
        
if __name__ =="__main__":

    rocks_MCQ()