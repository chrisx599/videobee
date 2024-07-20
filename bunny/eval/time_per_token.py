import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, tokenizer, image_processor, model_config, num_samples=64):
        self.image_paths = os.listdir(image_folder)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.questions = ["please describe the image." for _ in range(num_samples)]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        question_text = self.questions[index]
        question_text = DEFAULT_IMAGE_TOKEN + '\n' + question_text

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Prepare the image
        image = Image.open(os.path.join(self.image_folder, image_path)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # Prepare the text input
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.image_paths)


# DataLoader
def create_data_loader(image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    dataset = CustomDataset(image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    load_model_start_time = time.time()  # 开始计时
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type)
    load_model_end_time = time.time()  # 结束计时
    load_model_time = load_model_end_time - load_model_start_time  # 计算生成时间
    print('Time of loading the model:',load_model_time)

    total_average_time_per_token = 0  # 初始化总的平均时间变量
    total_generated_tokens = 0  # 初始化总生成token数量

    process_data_start_time = time.time()  # 开始计时
    data_loader = create_data_loader(args.image_folder, tokenizer, image_processor, model.config)
    process_model_end_time = time.time()  # 结束计时
    process_data_time = process_model_end_time - process_data_start_time  # 计算生成时间
    print('Time of loading the model:', process_data_time)

    for input_ids, image_tensor in tqdm(data_loader, total=len(data_loader.dataset)):

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        start_time = time.time()  # 开始计时

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=model.dtype, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        end_time = time.time()  # 结束计时
        generation_time = end_time - start_time  # 计算生成时间

        generated_tokens = output_ids[:, input_ids.shape[1]:]  # 获取仅生成的token部分
        num_generated_tokens = generated_tokens.shape[1]  # 计算生成的token数量

        if num_generated_tokens > 0:  # 避免除以零
            average_time_per_token = generation_time / num_generated_tokens
            total_average_time_per_token += average_time_per_token
            total_generated_tokens += 1

    if total_generated_tokens > 0:  # 确保不除以零
        grand_average_time_per_token = total_average_time_per_token / total_generated_tokens
        print(f"Grand average time per token for {total_generated_tokens} samples: {grand_average_time_per_token:.4f} seconds")
    else:
        print("No tokens were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default='phi-2')
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default='/zhaobai/DataOptim/siglip-so400m-patch14-384')
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
