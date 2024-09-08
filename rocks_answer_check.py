import asyncio
import logging
import os
import json
import re
from tqdm import tqdm
import replicate
import random
from openai import AsyncOpenAI,OpenAI
from zeno_build.models import lm_config

from utils import generate_from_openai_chat_completion,set_logger,NOISE_CHARS


def _context_for_1shot(prompt,que_list,ref_ans_list,url_list,):
    idx=random.randint(0,len(que_list)-1)
    user_example=[
        {
            "type": "text",
            "text": prompt+"\n"+que_list[idx]
        },
        {
            "type": "image_url",
            "image_url": {
                "url": url_list[idx],
            }
        }
    ]
    context=dict(messages=[
        {
            "content":user_example,
            "role":"user"
        },
        {
            "content":f"### 答案: {ref_ans_list[idx]}",
            "role":"assistant"
        }
    ])
    return context

async def _async_model_output_DEEPBRICKS(context_list):
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = ASYNC_MODEL_CONFIG, logger=None,)
    return raw_outputs


async def _async_model_output_RELICATE(context_list):
    
    raw_outputs=[]
    for context in tqdm(context_list):
        img_url=context["messages"][-1]["content"][1]["image_url"]["url"]
        text=context["messages"][-1]["content"][0]["text"]        
        
        output = replicate.run(
            model_name,
            input={
                "image": img_url,
                "top_p": 1,
                "prompt": text,
                "max_tokens": 4096,
                "temperature": 0.5
            }
        )
        raw_output=""
        for item in output:
            raw_output+=f"{item}"
        raw_outputs.append(raw_output)
        
    return raw_outputs


async def rocks_answer_check(prompt,data_file,res_file,one_shot_gate,logger=None,used_size="all",):
    data=json.load(open(data_file,"r"))
    if type(used_size) is int:
        data=data[:used_size]
        
    idx_list=[item["编号"] for item in data]
    que_list=[item["问题"] for item in data]
    ref_ans_list=[item["答案"] for item in data]
    url_list=[item["图片链接"]  for item in data]
    
    context_list=[]
    for que,url in zip(que_list,url_list):
        user_input=[
                {
                    "type": "text",
                    "text": prompt+"\n"+que
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    }
                }
            ]
    
        if one_shot_gate==1:
            context=_context_for_1shot(prompt,que_list,ref_ans_list,url_list,)
            context["messages"].append({
                    "content":user_input,
                    "role":"user"
                })
        else:
            context=dict(messages=[
                {
                    "content":user_input,
                    "role":"user"
                }
            ])
        
        context_list.append(context)
    
    if model_name in DEEP_BRICKS_MODEL_LIST:
        raw_outputs= await _async_model_output_DEEPBRICKS(context_list)
    elif model_name in REPLICATE_MODEL_LIST:
        raw_outputs= await _async_model_output_RELICATE(context_list)
    else:
        print("model not supported")
        exit()

    answer_check_list=[]
    correct_num=0
    for idx,que,url,ref_ans,raw_output in zip(idx_list,que_list,url_list,ref_ans_list,raw_outputs):
        print(raw_output)

        model_ans = raw_output.strip(NOISE_CHARS)
        match = re.search(r'[ABCD]', model_ans)
        if match:
            option=match.group(0)
            check=1 if option in ref_ans else 0
        else:
            check = 0

        if check == 1:
            correct_num+=1
        answer_check_list.append({
            "编号":idx,
            "问题":que,
            "图片链接":url,
            "参考答案":ref_ans,
            "模型输出":raw_output,
            "是否正确":check,
        })
    
    print(f"==================== 正确率：{correct_num}/{used_size} ====================")
    
    with open(res_file.replace(".json",".txt"), 'w',encoding='utf-8') as file:
        for d in answer_check_list:
            for key, value in d.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
            
    with open(res_file,"w",encoding='utf-8') as f:
        json.dump(answer_check_list,f,indent=4)
    
    
if __name__ =="__main__":
    API_KEYS=json.load(open("./const/api_key.json","r"))
    
    IMG_URL_PREFIX="https://huggingface.co/datasets/hexuan21/geo_proj/resolve/main/rocks"
    SOL_PROMPT=f"以下是大学地质学中的一道通过看图片判断岩石种类的题目,请直接输出正确选项的标号比如'A','B','C'等即可。\n\n输入格式:\n### 问题: <判断岩石种类>\n图片:<img, 岩石图片>\n输出格式: \n### 答案: <你选择的选项,如A>。\n以下是题目:"
    
    GPT_CHECK_PROMPT="这里有同一道题的两个答案,前一个是标准参考答案,后一个是学生写的答案。请判断学生的答案和参考答案相同是否符合,并给出评分(范围是0-1),\n\n输入格式:\n### 参考答案: <reference answer>\n### 学生的答案: <student's answer>\n输出格式: ### 判断：<float,1最高,表示完全正确,0最低,表示完全错误,打分的间隔是0.1>\n"
        
    DEEP_BRICKS_MODEL_LIST=["gpt-4o-2024-08-06","claude-3.5-sonnet",]
    REPLICATE_MODEL_LIST=[
    "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    "yorickvp/llava-v1.6-mistral-7b:19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874",
    "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
    "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
    "daanelson/minigpt-4:e447a8583cffd86ce3b93f9c2cd24f2eae603d99ace6afa94b33a08e94a3cd06",
    "lucataco/qwen-vl-chat:50881b153b4d5f72b3db697e2bbad23bb1277ab741c5b52d80cd6ee17ea660e9",
    "lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31",
    "cjwbw/internlm-xcomposer:d16df299dbe3454023fcb47ed48dbff052e9b7cdf2837707adff3581edd11e95",
    "adirik/owlvit-base-patch32:5e899f155a1913c4b7304d09082d842ca7fe6cb1f22e066c83eb1d7849dc37c2",
    "lucataco/moondream1:ecd26482e4c9220957e22290cb616200b51217fe807f61653a8459ed7541e9d5",
    "adirik/kosmos-g:56f9fde586eeecfd03c9c34da1c40f5e513af2d511d4b1961f810df1334cc6e9",
    "adirik/masactrl-sdxl:cb949d990cba61f0ecc242912f4305b1780e552f239d21930f4eb24e713bb599",
    "zsxkib/uform-gen:e6fa8e2d076907b45a0b535a14ddb22402548c2e478310cd18daa1c4c01f422b",
    "cjwbw/unidiffuser:0967db6b8843a90b4b1a2be08f553fcba60dbb8790c79864469644b1c8eecfd7",
    "cjwbw/unival:00a9af2b0889db2a73f5002b3a3000ba4eec8c5fed0cb4fd842a3fc75ab0e98f",
                          ]

    # model_name="gpt-4o-2024-08-06"
    model_name="claude-3.5-sonnet"
    
    # model_name="yorickvp/llava-v1.6-mistral-7b:19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874"
    # model_name="yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"
    # model_name="yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63"
    # model_name="yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174"
    # model_name="lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31"
    # model_name="lucataco/moondream1:ecd26482e4c9220957e22290cb616200b51217fe807f61653a8459ed7541e9d5"
    
    # error
    # model_name="zsxkib/uform-gen:e6fa8e2d076907b45a0b535a14ddb22402548c2e478310cd18daa1c4c01f422b"
    # model_name="cjwbw/unidiffuser:0967db6b8843a90b4b1a2be08f553fcba60dbb8790c79864469644b1c8eecfd7"
    # model_name="cjwbw/unival:00a9af2b0889db2a73f5002b3a3000ba4eec8c5fed0cb4fd842a3fc75ab0e98f"
    # model_name="adirik/owlvit-base-patch32:5e899f155a1913c4b7304d09082d842ca7fe6cb1f22e066c83eb1d7849dc37c2"
    # model_name="adirik/masactrl-sdxl:cb949d990cba61f0ecc242912f4305b1780e552f239d21930f4eb24e713bb599"
    # model_name="adirik/kosmos-g:56f9fde586eeecfd03c9c34da1c40f5e513af2d511d4b1961f810df1334cc6e9"
    # model_name="lucataco/qwen-vl-chat:50881b153b4d5f72b3db697e2bbad23bb1277ab741c5b52d80cd6ee17ea660e9"
    # model_name="daanelson/minigpt-4:e447a8583cffd86ce3b93f9c2cd24f2eae603d99ace6afa94b33a08e94a3cd06"
    # model_name="cjwbw/internlm-xcomposer:d16df299dbe3454023fcb47ed48dbff052e9b7cdf2837707adff3581edd11e95"
    
    if model_name in DEEP_BRICKS_MODEL_LIST:
        os.environ["OPENAI_API_KEY"]=API_KEYS["OpenAI_API_KEYd1"]
        os.environ["OPENAI_ORG"]=""
        os.environ["OPENAI_BASE_URL"]=API_KEYS["OpenAI_BASE_URL"]
    elif model_name in REPLICATE_MODEL_LIST:
        os.environ["REPLICATE_API_TOKEN"]=API_KEYS["REPLICATE_API_TOKEN"]
    else:
        print("model not supported")
        exit()

    
    ASYNC_MODEL_CONFIG = lm_config.LMConfig(provider="openai_chat", model=model_name)
    
    one_shot_gate=0
    data_file="./data/rocks_MCQ.json"
    if "/" in model_name or ":" in model_name:
        model_name_simple=model_name.split(":")[0].replace("/","@")
    else:
        model_name_simple=model_name
    res_file=f"./res_rocks/rocks_{model_name_simple}.json"
    if one_shot_gate==1:
        res_file=f"./res_rocks/rocks_{model_name_simple}_1shot.json"
    used_size="all"
    
    logger_file="./logs/rocks_answer_check.log"
    os.makedirs(os.path.dirname(res_file),exist_ok=True)
    
    # logger=set_logger(logger_file)
    asyncio.run(rocks_answer_check(prompt=SOL_PROMPT,data_file=data_file,res_file=res_file,used_size=used_size,one_shot_gate=one_shot_gate))
    
    
