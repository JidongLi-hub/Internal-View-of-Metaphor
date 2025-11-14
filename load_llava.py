import sys
sys.path.append("/model/fangly/mllm/ljd/LLaVA-NeXT/")
sys.path.append("/model/fangly/mllm/ljd/LLaVA-NeXT/llava/")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base,model_name,device_map="auto",attn_implementation=None)
    return tokenizer, model, image_processor