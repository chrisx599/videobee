import os
import io
import json
import re
import torch
import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,MULTI_IMAGE_TOKEN_1,MULTI_IMAGE_TOKEN_2,MULTI_IMAGE_TOKEN_3,MULTI_IMAGE_TOKEN_4,MULTI_IMAGE_TOKEN_5,MULTI_IMAGE_TOKEN_6,MULTI_IMAGE_TOKEN_7,MULTI_IMAGE_TOKEN_8
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import process_images,process_videos, tokenizer_image_token,tokenizer_multi_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from bunny.util.video_transforms import GroupNormalize, GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret


class MLVU(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        # torch_imgs = self.transform(images_group)
        # return torch_imgs
        return images_group
    
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
            'video_path':video_path
        }


def load_model(args):
     # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, video_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type)

    return tokenizer, model, image_processor, video_processor, context_len


def eval_model(tokenizer, model, video_processor, context_len,data):
    print("#################")
    print("question",data['question'])
    print("answer",data['answer'])
    image=data['video']
    question=data['question']
    answer=data['answer']
    token_idx=""
    token_idx_list=[]
    image_idx=-201
    for i in range(1, 17):
        token_idx += f"<image {i}>"
        token_idx_list.append(image_idx)
        image_idx=-201-i
    # 添加额外的文本
    qs = token_idx + '\n' + question + '\n' + "Only give the best option."
   
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], "Best Option: (")
    # prompt = conv.get_prompt()
    prompt = get_prompt2(conv)
    input_ids = tokenizer_multi_image_token(prompt, tokenizer, image_token_index=token_idx_list, return_tensors='pt').unsqueeze(0).to(model.device)
    image_tensor = process_videos(image, video_processor, model.config)
    image_tensor = torch.unbind(image_tensor, dim=1)
    image_tensor=list(image_tensor)
    print("$$$$$$$$",len(image_tensor))
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=128,
            use_cache=True)
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print("outputs",outputs)
    return outputs


def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[:index3]

    if pred==gt_option:
        print("11111111111111",pred,gt_option)
        flag=True

    return flag



if __name__ == "__main__":
    num_frame =16
    resolution = 384
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/zhaobai46d/share/shuyan/videobunny/checkpoints-phi-3/bunny-lora-phi-3-imagewvideo1")
    parser.add_argument("--model-base", type=str, default="/zhaobai46d/share/bunny/models/Phi-3-mini-128k-instruct")
    parser.add_argument("--model-type", type=str, default="phi-3")
    parser.add_argument("--conv-mode", type=str, default="phi3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--single-pred-prompt", action="store_true")

    args = parser.parse_args()
 

    
    data_list = {
    "count": ("4_count.json", f"/zhaobai46d/share/shuyan/MLVU/video/4_count", "video"),
    "ego": ("3_ego.json", f"/zhaobai46d/share/shuyan/MLVU/video/3_ego", "video"),
    "needle": ("2_needle.json", f"/zhaobai46d/share/shuyan/MLVU/video/2_needle", "video"),
    "order": ("5_order.json", f"/zhaobai46d/share/shuyan/MLVU/video/5_order", "video"),
    "plotQA": ("1_plotQA.json", f"/zhaobai46d/share/shuyan/MLVU/video/1_plotQA", "video"),
    "anomaly_reco": ("6_anomaly_reco.json", f"/zhaobai46d/share/shuyan/MLVU/video/6_anomaly_reco", "video"),
    "topic_reasoning": ("7_topic_reasoning.json", f"/zhaobai46d/share/shuyan/MLVU/video/7_topic_reasoning", "video")
    }

    data_dir = f"/zhaobai46d/share/shuyan/MLVU/json"
    dataset = MLVU(data_dir, data_list, num_segments=num_frame, resolution=resolution)
    ## load model
    tokenizer, model, image_processor, video_processor, context_len = load_model(args)
    
    
    save_path = "./test"
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    for example in tqdm(dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred=eval_model(tokenizer, model, video_processor, context_len, example)
        gt = example['answer']
        
        
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    total=0
    idx=0
    for k, v in acc_dict.items():
        idx+=1
        final_res[k] = v[0] / v[1] * 100
        total+=final_res[k]

    final_res['Avg'] = total /idx 
    print(final_res)

    with open("mlvu_bench.json", "w") as f:
        json.dump(final_res, f)
