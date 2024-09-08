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


def _score_two_ans(check_prompt,ref_ans,model_ans):
    
    user_input=check_prompt+"\n### 参考答案:\n"+ref_ans+"\n### 学生的答案:\n"+model_ans
    response = client_judge.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": user_input},
        ],
        stream=False
    )
    raw_output=response.choices[0].message.content
    
    return raw_output




async def _model_output_DEEPSEEK(context_list):
    raw_outputs=[]
    for context in tqdm(context_list):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": context["messages"][0]["content"]},
            ],
            stream=False
        )
        raw_output=response.choices[0].message.content
        raw_outputs.append(raw_output)
    return raw_outputs


async def _async_model_output_DEEPBRICKS(context_list):
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = ASYNC_MODEL_CONFIG, logger=None,)
    return raw_outputs


async def _model_output_REPLICATE(context_list):
    raw_outputs=[]
    for context in tqdm(context_list):
        text=context["messages"][0]["content"]       
        
        output = replicate.run(
            model_name,
            input={
                "prompt": text,
                "max_length": 200
            }
        )
        raw_output=""
        for item in output:
            raw_output+=f"{item}"
        raw_outputs.append(raw_output)
        
    return raw_outputs


async def gen_and_check_async(prompt,data_file,check_res_file,MCQ_or_QA,used_size="all",):
    
    data=json.load(open(data_file,"r"))
    if type(used_size) is int:
        data=data[:used_size]
        
    idx_list=[item["编号"] for item in data]
    que_list=[item["问题"] for item in data]
    ref_ans_list=[item["答案"] for item in data]
    
    context_list=[]
    for que in que_list:
        user_input=prompt+"\n### 问题: \n"+que
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)

    if model_name in DEEP_BRICKS_MODEL_LIST:
        raw_outputs = await _async_model_output_DEEPBRICKS(context_list)
    elif model_name in DEEP_SEEK_MODEL_LIST:
        raw_outputs = await _model_output_DEEPSEEK(context_list)
    elif model_name in REPLICATE_MODEL_LIST:
        raw_outputs = await _model_output_REPLICATE(context_list)
    else:
        print("model not supported")
        exit()
        
    
    answer_check_list=[]
    correct_num=0
    for idx,que,ref_ans,raw_output in tqdm(zip(idx_list,que_list,ref_ans_list,raw_outputs)):
        # print(raw_output)
        if MCQ_or_QA==1:
            check_detail =_score_two_ans(GPT_CHECK_PROMPT_FLOAT,ref_ans,raw_output)
            matches = re.findall(r'\d+\.\d+|\d+', check_detail)
            score = float(matches[0]) if matches else 0.0
        else:
            check_detail = _score_two_ans(GPT_CHECK_PROMPT_BINARY,ref_ans,raw_output)
            score=1 if "1" in check_detail else 0
        
        if type(score) is int and score==1:
            correct_num+=1
        
        answer_check_list.append({
            "---编号---":idx,
            "---问题---":que,
            "---参考答案---":ref_ans,
            "---模型输出---":raw_output,
            "---机器检查---":check_detail,
            "---得分---":score,
        })
    
    with open(res_file.replace(".json",".txt"), 'w',encoding='utf-8') as file:
        for d in answer_check_list:
            for key, value in d.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
    with open(check_res_file,"w",encoding='utf-8') as f:
        json.dump(answer_check_list,f,indent=4)
    
    if MCQ_or_QA==0:
        print(f"==================== MCQ的正确率:{correct_num}/{used_size} ====================")
    else:
        check_score_list=[float(item["得分"]) for item in answer_check_list]
        avg_score=sum(check_score_list)/len(check_score_list)
        final_avg_score=round(avg_score*100,2)
        print(f"==================== QA的平均得分:{final_avg_score} ====================")

    
if __name__ == "__main__":
    API_KEYS=json.load(open("./const/api_key.json","r"))
    
    SOL_PROMPT=f"以下是大学地质学中的一道选择题或者填空题,请给出正确的回答。对于选择题,直接输出正确选项的标号比如'A','B','C','D'即可; 对于简答题,注意答案应该在50-100字。\n\n输入格式:\n### 问题: <一道地质学选择题或简答题>\n输出格式: \n### 答案: <你输出的答案>。以下是题目:"
    GPT_CHECK_PROMPT_FLOAT="这里有同一道题的两个答案,前一个是标准参考答案,后一个是学生写的答案。请判断学生的答案和参考答案是否符合,并给出评分(范围是0-1),\n\n输入格式:\n### 参考答案: <reference answer>\n### 学生的答案: <student's answer>\n输出格式: ### 判断：<float,1最高,表示学生的答案非常符合参考答案,0最低,表示学生的答案完全错误,打分的间隔是0.1>\n"
    
    GPT_CHECK_PROMPT_BINARY="这里有同一道单项选择题的两个答案,前一个是标准参考答案,后一个是学生写的答案。请判断学生的选项和参考答案的正确选项是否一样,并给出判断(0表示错误, 1表示正确)。不论是中文还是英文,只要学生的答案选出来了正确选项，那就是对的。\n\n输入格式:\n### 参考答案: <reference answer>\n### 学生的答案: <student's answer>\n输出格式: ### 判断：<int,0表示错误, 1表示正确>\n"
    
    DEEP_BRICKS_MODEL_LIST=["gpt-4o-2024-08-06","gpt-4o-mini","gpt-4o","gpt-4-turbo","claude-3.5-sonnet","llama-3.1-405b","llama-3.1-70b","llama-3-70b"]
    DEEP_SEEK_MODEL_LIST=["deepseek-coder","deepseek-chat"]
    REPLICATE_MODEL_LIST=[
        "meta/llama-2-7b-chat",
    "meta/llama-2-13b-chat",
    "meta/llama-2-70b-chat",
    "replicate/llama-7b:03d3a482ec4f2ec1809171d0ffbd3be7d2a775a01c6bfb5988f4acf39d64f0ce",
    "mistralai/mistral-7b-v0.1",
    "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "01-ai/yi-6b:d302e64fad6b4d85d47b3d1ed569b06107504f5717ee1ec12136987bec1e94f1",
    "01-ai/yi-6b-chat:14efadfaf772f45ee74c14973007cbafab3ccac90169ec96a9fc7a804253535d",
    "01-ai/yi-34b-chat:914692bbe8a8e2b91a4e44203e70d170c9c5ccc1359b283c84b0ec8d47819a46",
    "replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
    "stability-ai/stablelm-tuned-alpha-7b:943c4afb4d0273cf1cf17c1070e182c903a9fe6b372df36b5447cf45935c42f2",
    "google-deepmind/gemma-2b:26b2c530f16236a4816611509730c2e6f7b27875a6d33ec5cff42961750c98d8",
    "google-deepmind/gemma-2b-it:dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626",
    "google-deepmind/gemma-7b:2ca65f463a2c0cfef4dbc4ba70d227ed96455ef6020c1f6983b2a4c4f3ecb4ec",
    "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
    "lucataco/qwen1.5-7b:f85bec5b21ba0860e0f200be6ef5af9d5a65b974b9f99e36eb036d21eab884de",
    "lucataco/qwen1.5-14b:28c4bbc17ee1575bd2efe2d805a6c3da9f555bf6298d447d9d8d8ebfb891c4a1",
    "lucataco/qwen1.5-72b:f919d3c43a8758de744cf2908426dd744154120f0a22e457a3fa647acdfe33be",
    "lucataco/phi-2:740618b0c24c0ea4ce5f49fcfef02fcd0bdd6a9f1b0c5e7c02ad78e9b3b190a6",
    "lucataco/phixtral-2x2_8:25d7b93bb0ec9e8dd94fcc69adc786759243a5628ba5574bd9609d6abafe57cf",
    "nateraw/nous-hermes-llama2-awq:b3f3f0120a3c4fd37a5e75164cc3ed883c248b9e6d004a70f0d31c3b0debb604",
    "nateraw/nous-hermes-2-solar-10.7b:1e918ab6ffd5872c21fba21a511f344fd12ac0edff6302c9cd260395c7707ff4",
    "kcaverly/nous-hermes-2-yi-34b-gguf:5e61add6304f8a7bffda9089153f8af3fd559cb9ad783f73703fd9ddcc9e8fde",
    "adirik/mamba-130m:d7fe29cf1ad11cbd2fc28b808a7a7b4fd1ff880c394eaef914e40b6da08934ce",
    "adirik/mamba-370m:e6a271d772f0eef0703af2d246e45bace40ce365e04a7bf464af66bf548852c7",
    "adirik/mamba-790m:77782448285ebc03a24c2e90cc12b6cebbdaf325c071eaee2e315320308d9748",
    "adirik/mamba-1.4b:917b6c8e21c963b41147b2f232f8e5f3c2f572e54f63897ed5dc6793a9972f5d",
    "adirik/mamba-2.8b-slimpj:e663586c1922943ad155a146b4c843855e61241d9ac80db6103215927e5134ed",
    "adirik/mamba-2.8b:571abd73203a3dd3d7071f1c0380a3502c427aba98a2fb5edf2f7cfdeea1676c",
    ]
    
    # model_name="gpt-4o-2024-08-06"
    # model_name="gpt-4o-mini"
    # model_name="claude-3.5-sonnet"
    # model_name="llama-3.1-405b"
    # model_name="llama-3.1-70b"
    # model_name="llama-3-70b"
    # model_name="deepseek-chat"
    # model_name="deepseek-coder"
    # model_name="meta/llama-2-7b-chat"
    # model_name="meta/llama-2-13b-chat"
    # model_name="mistralai/mistral-7b-v0.1"
    # model_name="mistralai/mistral-7b-instruct-v0.2"
    # model_name="mistralai/mixtral-8x7b-instruct-v0.1"
    # model_name="01-ai/yi-6b:d302e64fad6b4d85d47b3d1ed569b06107504f5717ee1ec12136987bec1e94f1"
    # model_name="01-ai/yi-6b-chat:14efadfaf772f45ee74c14973007cbafab3ccac90169ec96a9fc7a804253535d"
    # model_name="01-ai/yi-34b-chat:914692bbe8a8e2b91a4e44203e70d170c9c5ccc1359b283c84b0ec8d47819a46"
    # model_name="replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210"
    # model_name="stability-ai/stablelm-tuned-alpha-7b:943c4afb4d0273cf1cf17c1070e182c903a9fe6b372df36b5447cf45935c42f2"
    # model_name="google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5"
    # model_name="lucataco/qwen1.5-7b:f85bec5b21ba0860e0f200be6ef5af9d5a65b974b9f99e36eb036d21eab884de"
    
    model_name="lucataco/qwen1.5-14b:28c4bbc17ee1575bd2efe2d805a6c3da9f555bf6298d447d9d8d8ebfb891c4a1"
    
    # model_name="lucataco/qwen1.5-72b:f919d3c43a8758de744cf2908426dd744154120f0a22e457a3fa647acdfe33be"
    # model_name="nateraw/nous-hermes-llama2-awq:b3f3f0120a3c4fd37a5e75164cc3ed883c248b9e6d004a70f0d31c3b0debb604"
    # model_name="nateraw/nous-hermes-2-solar-10.7b:1e918ab6ffd5872c21fba21a511f344fd12ac0edff6302c9cd260395c7707ff4"
    # model_name="kcaverly/nous-hermes-2-yi-34b-gguf:5e61add6304f8a7bffda9089153f8af3fd559cb9ad783f73703fd9ddcc9e8fde"
    # model_name="adirik/mamba-130m:d7fe29cf1ad11cbd2fc28b808a7a7b4fd1ff880c394eaef914e40b6da08934ce"
    # model_name="adirik/mamba-370m:e6a271d772f0eef0703af2d246e45bace40ce365e04a7bf464af66bf548852c7"
    # model_name="adirik/mamba-790m:77782448285ebc03a24c2e90cc12b6cebbdaf325c071eaee2e315320308d9748"
    # model_name="adirik/mamba-1.4b:917b6c8e21c963b41147b2f232f8e5f3c2f572e54f63897ed5dc6793a9972f5d"
    # model_name="adirik/mamba-2.8b-slimpj:e663586c1922943ad155a146b4c843855e61241d9ac80db6103215927e5134ed"
    # model_name="adirik/mamba-2.8b:571abd73203a3dd3d7071f1c0380a3502c427aba98a2fb5edf2f7cfdeea1676c"
    # model_name="lucataco/phi-2:740618b0c24c0ea4ce5f49fcfef02fcd0bdd6a9f1b0c5e7c02ad78e9b3b190a6"
    # model_name="lucataco/phixtral-2x2_8:25d7b93bb0ec9e8dd94fcc69adc786759243a5628ba5574bd9609d6abafe57cf"

    # model_name="meta/llama-2-70b-chat"
    # model_name="replicate/llama-7b:03d3a482ec4f2ec1809171d0ffbd3be7d2a775a01c6bfb5988f4acf39d64f0ce"
    # model_name="google-deepmind/gemma-2b:26b2c530f16236a4816611509730c2e6f7b27875a6d33ec5cff42961750c98d8"
    # model_name="google-deepmind/gemma-2b-it:dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626"
    # model_name="google-deepmind/gemma-7b:2ca65f463a2c0cfef4dbc4ba70d227ed96455ef6020c1f6983b2a4c4f3ecb4ec"
    
    if model_name in DEEP_BRICKS_MODEL_LIST:
        os.environ["OPENAI_API_KEY"]=API_KEYS["OpenAI_API_KEYd1"]
        os.environ["OPENAI_ORG"]=""
        os.environ["OPENAI_BASE_URL"]=API_KEYS["OpenAI_BASE_URL"]
    elif model_name in DEEP_SEEK_MODEL_LIST:
        os.environ["OPENAI_API_KEY"]=API_KEYS["DeepSeek_API_KEY"]
        os.environ["OPENAI_ORG"]=""
        os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepSeek_BASE_URL"]
    elif model_name in REPLICATE_MODEL_LIST:
        os.environ["REPLICATE_API_TOKEN"]=API_KEYS["REPLICATE_API_TOKEN"]
    else:
        print("model not supported")
        exit()
    
    ASYNC_MODEL_CONFIG = lm_config.LMConfig(provider="openai_chat", model=model_name)
    MCQ_or_QA=0
    
    if "/" in model_name or ":" in model_name:
        model_name_simple=model_name.split(":")[0].replace("/","@")
    else:
        model_name_simple=model_name
    
    if MCQ_or_QA==0:
        data_file="./data/geo_MCQ.json"
        res_file=f"./res_knowledge_MCQ/knowledge_{model_name_simple}.json"
    else:
        data_file="./data/geo_QA.json"
        res_file=f"./res_knowledge_QA/knowledge_{model_name_simple}.json"
    used_size=50
    os.makedirs(os.path.dirname(res_file),exist_ok=True)
    
    logger_file="./logs/knowledge_answer_check.log"
    # logger=set_logger(logger_file)
    
    client=None
    if model_name in DEEP_BRICKS_MODEL_LIST or model_name in DEEP_SEEK_MODEL_LIST:
        client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"])
    
    JUDGE_MODEL_NAME="gpt-4o-2024-08-06"
    client_judge = OpenAI(base_url=API_KEYS["OpenAI_BASE_URL"],api_key=API_KEYS["OpenAI_API_KEYd1"])
    
    asyncio.run(gen_and_check_async(prompt=SOL_PROMPT,data_file=data_file,check_res_file=res_file,MCQ_or_QA=MCQ_or_QA,used_size=used_size,))
    