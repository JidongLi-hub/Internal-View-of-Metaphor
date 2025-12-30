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