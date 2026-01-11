import torch
from torch import nn
from torch import tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import requests
from vllm import LLM, SamplingParams
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap
import pandas as pd
import re

def inspect_df(df, sample_n=3):
    """parquet文件的基本信息检查函数"""
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head(sample_n))
    print("\nRandom Sample:")
    print(df.sample(sample_n, random_state=0))
    print("\nNull Ratio:")
    print(df.isnull().mean().sort_values(ascending=False))


MLLM_client = OpenAI(
                    base_url="http://localhost:7777/v1",
                    api_key="00000000",
                )

LLM_client = OpenAI(
                    base_url="http://localhost:8888/v1",
                    api_key="00000000",
                )
class MLLM:
    def __init__(self, client=MLLM_client, model_name="/data/models/Qwen2.5-VL-72B-Instruct"):
        self.model_name = model_name
        self.client = client
    
    def chat(self, image_path=None, text="Hello!"):
        if image_path is not None:
            content = [{"type": "image_url", "image_url": {"url": "file://"+image_path}},
                                            {"type": "text", "text":text
                                }] 
        else:
            content = text
        completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {   "role": "user", 
                                "content": content
                            }
                        ]
                    )
        return completion.choices[0].message.content.strip()

class LLM:
    def __init__(self, client=LLM_client, model_name="/data/models/Qwen-Qwen3-32B"):
        self.client = client
        self.model_name = model_name

    def chat(self, text):
        completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": text}
                ],
                extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},  # 关闭Qwen3的思考模式
                    },
                )
        
        return completion.choices[0].message.content.strip()


    # 发送http请求时要带上Bearer认证
    # headers = {"Authorization": "Bearer 00000000"}
    # response = requests.get(url="http://localhost:7777/v1/models", headers=headers)
    # print(response.json())