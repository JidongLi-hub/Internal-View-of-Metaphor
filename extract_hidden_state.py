from utils import *
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

def load_qwen2_vl_model(model_path="/data/fangly/mllm/ljd/models/Qwen2.5-VL-7B-Instruct"):
    """加载Qwen2.5-VL-7B-Instruct模型"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 避免溢出
            device_map='auto',
            attn_implementation="eager"  
        ) 
    processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels = 256*28*28,
            max_pixels = 480*28*28
        )
    processor.tokenizer.padding_side = "left"
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return model, processor, tokenizer


def parse_full_output(full_output: str) -> str:
    """
    从full_output中提取user部分的问题内容（不包含答案）
    
    full_output格式示例:
    "system\nYou are a helpful assistant.\nuser\nYou are given a meme.\n...## Answer: \n\nassistant\nAC"
    
    我们需要提取 user 后面到 ## Answer: 之前的内容
    """
    # 方法1: 使用正则提取user和assistant之间的内容
    # 匹配 "user\n" 到 "## Answer:" 之间的内容
    pattern = r"user\n(.*?)## Answer:"
    match = re.search(pattern, full_output, flags=re.DOTALL)
    
    if match:
        user_content = match.group(1).strip()
        # 加上 ## Answer: 作为结尾，让模型回答
        return user_content + "\n## Answer:"
    else:
        # 如果正则匹配失败，尝试简单分割
        if "## Answer:" in full_output:
            parts = full_output.split("## Answer:")
            # 找到user部分
            if "user\n" in parts[0]:
                user_part = parts[0].split("user\n", 1)[-1]
                return user_part.strip() + "\n## Answer:"
    
    # 如果都失败，返回原始内容（去掉assistant回答部分）
    print(f"Warning: Could not parse full_output properly")
    return full_output.split("assistant")[0].strip()


def extract_last_token_hidden_state(
        model,
        processor, 
        dataset,
        image_dir="/model/fangly/mllm/ljd/memeqa/data/semeval_img",
        target_layers=[-1, -2],
        output_dir="data/hidden_states"
    ):
    """
    抽取指定层的最后一个token的hidden state作为分类器的输入特征
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每一层准备存储列表
    hidden_states_by_layer = {layer: [] for layer in target_layers}
    labels = []
    
    # 标签映射
    label_map = {
        "negative": 0,
        "positive": 1
    }
    
    model.eval()
    
    # 用于统计
    nan_count = 0
    skipped_count = 0
    
    for item in tqdm(dataset, desc="Extracting hidden states"):
        image_id = item["image_id"]
        full_output = item["full_output"]
        label = item["label"]
        
        # 检查标签是否有效
        if label not in label_map:
            print(f"Warning: Unknown label '{label}' for {image_id}, skipping...")
            skipped_count += 1
            continue
        
        # 解析full_output，提取问题部分
        question_text = parse_full_output(full_output)
        
        # 构建图片路径
        image_path = os.path.join(image_dir, image_id)
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            skipped_count += 1
            continue
        
        # 构建Qwen2.5-VL的消息格式 - 正确的格式
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": question_text
                    }
                ]
            }
        ]
        
        try:
            # 处理输入
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
            # 添加调试信息
            all_hidden_states = outputs.hidden_states
            print(f"Number of hidden state layers: {len(all_hidden_states)}")
            print(f"Last layer hidden state dtype: {all_hidden_states[-1].dtype}")
            print(f"Last layer hidden state shape: {all_hidden_states[-1].shape}")
            
            # 检查原始数据是否就有 NaN
            for i, hs in enumerate(all_hidden_states[-3:]):
                has_nan = torch.isnan(hs).any().item()
                has_inf = torch.isinf(hs).any().item()
                print(f"Layer {i-3}: NaN={has_nan}, Inf={has_inf}, "
                      f"min={hs.min().item():.4f}, max={hs.max().item():.4f}")

            # 提取指定层的最后一个token的hidden state
            all_hidden_states = outputs.hidden_states
            
            # 检查是否有NaN
            has_nan = False
            temp_hidden_states = {}
            
            for layer_idx in target_layers:
                layer_hidden = all_hidden_states[layer_idx]
                # 先在 GPU 上检查，避免不必要的数据传输
                if torch.isnan(layer_hidden).any() or torch.isinf(layer_hidden).any():
                    has_nan = True
                    nan_count += 1
                    print(f"Warning: NaN/Inf detected in hidden state for {image_id}")
                    print(f"  Layer {layer_idx}: min={layer_hidden.min()}, max={layer_hidden.max()}")
                    break
                
                # 转换为 float32 再转 numpy
                last_token_hidden = layer_hidden[0, -1, :].float().cpu().numpy()
                temp_hidden_states[layer_idx] = last_token_hidden
            
            if has_nan:
                skipped_count += 1
                continue
            
            # 如果没有NaN，保存结果
            for layer_idx in target_layers:
                hidden_states_by_layer[layer_idx].append(temp_hidden_states[layer_idx])
            
            labels.append(label_map[label])
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            skipped_count += 1
            continue
        
        finally:
            # 释放内存
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
    
    print(f"\nExtraction completed:")
    print(f"  - Total processed: {len(labels)}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - NaN samples: {nan_count}")
    
    # 检查标签分布
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  - Label distribution: {dict(zip(unique, counts))}")
    
    if len(labels) == 0:
        print("Error: No valid samples extracted!")
        return None, None
    
    # 为每一层保存一个文件
    for layer_idx in target_layers:
        hidden_states = np.array(hidden_states_by_layer[layer_idx], dtype=np.float32)
        
        # 最终检查
        print(f"\nLayer {layer_idx} stats:")
        print(f"  - Shape: {hidden_states.shape}")
        print(f"  - NaN count: {np.isnan(hidden_states).sum()}")
        print(f"  - Inf count: {np.isinf(hidden_states).sum()}")
        print(f"  - Min: {np.min(hidden_states):.4f}, Max: {np.max(hidden_states):.4f}")
        print(f"  - Mean: {np.mean(hidden_states):.4f}, Std: {np.std(hidden_states):.4f}")
        
        # 命名文件
        if layer_idx < 0:
            layer_name = f"layer_minus_{abs(layer_idx)}"
        else:
            layer_name = f"layer_{layer_idx}"
        
        output_file = os.path.join(output_dir, f"hidden_states_{layer_name}.npz")
        
        np.savez(
            output_file,
            hidden_states=hidden_states,
            labels=labels
        )
        print(f"Saved to {output_file}")
    
    return hidden_states_by_layer, labels


def main():
    
    # 加载模型
    model_path = "/data/fangly/mllm/ljd/models/Qwen2.5-VL-7B-Instruct"  
    model, processor, tokenizer = load_qwen2_vl_model(model_path)
    
    # 加载数据集
    dataset_path = "data/P&N_Qwen.json"  
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 打印数据集信息
    print(f"Dataset size: {len(dataset)}")
    # 打印标签分布
    label_counts = {}
    for item in dataset:
        label = item.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution in dataset: {label_counts}")
    
    # 打印一个样本看看格式
    print("\nSample full_output:")
    print(dataset[0]["full_output"][:500])
    print("...")
    
    # 图片目录
    image_dir = "/model/fangly/mllm/ljd/memeqa/data/semeval_img"  
    
    # 提取hidden states
    target_layers = [-5, -6, -10, -11,  -14, -15, -16, -17, -18]
    
    extract_last_token_hidden_state(
        model=model,
        processor=processor,
        dataset=dataset,
        image_dir=image_dir,
        target_layers=target_layers,
        output_dir="data/hidden_states"
    )


if __name__ == "__main__":
    main()