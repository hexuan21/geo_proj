# -*- coding: utf-8 -*-
import asyncio
import logging
import os
import json
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import random

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from zeno_build.models import lm_config

from utils import generate_from_openai_chat_completion,set_logger,NOISE_CHARS


def remove_empty_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    non_empty_lines = [line for line in lines if line.strip() != '']
    # with open(file_path, 'w', encoding='utf-8') as file:
    #     file.writelines(non_empty_lines)
    return non_empty_lines


async def draw_qa(prompt,model_config,raw_file,dest_file,used_size="all",logger=None):
    lines=remove_empty_lines(raw_file)
    if type(used_size) is int:
        lines=lines[:used_size]
    
    chunk_list=['\n'.join(lines[i:i+CHUNK_LEN]) for i in range(0, len(lines), CHUNK_LEN)]
    chunk_list+=['\n'.join(lines[i:i+CHUNK_LEN]) for i in range(int(CHUNK_LEN/2), len(lines), CHUNK_LEN)]
    random.shuffle(chunk_list)
    
    context_list=[]
    for chunk in zip(chunk_list):
        user_input=prompt+"\n### 一段复习提纲: \n"
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = model_config, logger=logger,)

    qa_list=[]
    idx=0
    for raw_output in raw_outputs:
        if "N/A" in raw_output:
            continue
        sign_1="### 问题"
        sign_2="### 答案"
        if sign_1 in raw_output and sign_2 in raw_output:
            que = raw_output.split(sign_1)[1].split(sign_2)[0].strip(NOISE_CHARS)
            sol=raw_output.split(sign_2)[1].strip(NOISE_CHARS)
        else:
            continue
        qa_list.append({
            "编号":f"{idx:05d}",
            "问题":que,
            "答案":sol,
        })
        idx+=1
    with open(dest_file.replace(".json",".txt"), 'w',encoding='utf-8') as file:
        for d in qa_list:
            for key, value in d.items():
                file.write(f"{key}: {value}")
                if key=="答案":
                    file.write("\n")
            file.write("\n")
    with open(dest_file,"w") as f:
        json.dump(qa_list,f,indent=4)

    
if __name__ == "__main__":
    API_KEYS=json.load(open("./const/api_key.json","r"))
    os.environ["OPENAI_API_KEY"]=API_KEYS["OpenAI_API_KEYd1"]
    # os.environ["OPENAI_ORG"]=API_KEYS["OpenAI_ORG_ID"]
    os.environ["OPENAI_ORG"]=""
    os.environ["OPENAI_BASE_URL"]=API_KEYS["OpenAI_BASE_URL"]
    
    model_name="gpt-4o-2024-08-06"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)
    
    CHUNK_LEN=10
    
    MCQ_PROMPT=f"以下是地质学复习提纲中的一部分,将根据这段提纲生成一道有四个选项的单项选择题。\n确保生成的问题长度在50-100字,每个选项在50-100字,不要太短也不要太长。你应该尽量生成难的题目。\n\n输入格式:\n### 一段复习提纲:<复习提纲> \n输出格式:\n### 问题:<你生成的问题>\n### 答案:<你所生成的问题的答案>。"
    
    QA_PROMPT=f"以下是地质学复习提纲中的一部分,将根据这段提纲生成一道简答题。\n确保生成的问题长度在50-100字,答案在100-200字,不要太短也不要太长。你应该尽量生成难的题目。\n\n输入格式:\n### 一段复习提纲:<复习提纲> \n输出格式:\n### 问题:<你生成的问题>\n### 答案:<你所生成的问题的答案>。"
    
    prompt=QA_PROMPT
    logger_file="./logs/draw_qa.log"
    raw_file="./data/raw/raw_all_1.txt"
    dest_file="./data/geo_QA.json"
    os.makedirs(os.path.dirname(dest_file),exist_ok=True)
    
    # logger=set_logger(logger_file)
    asyncio.run(draw_qa(prompt=prompt,model_config=model_config,raw_file=raw_file,dest_file=dest_file))