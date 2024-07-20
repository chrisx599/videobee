import ast
import base64
import glob
from io import BytesIO
import os
import copy
from dataclasses import dataclass, field
import json
import random
from typing import Dict, Sequence, Optional, List
import re
import torch
import decord
import transformers
import numpy as np
from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN,MAX_IMAGE_LENGTH,IMAGE_TOKEN_INDEX
from torch.utils.data import Dataset

from bunny import conversation as conversation_lib
from bunny.util.mm_utils import tokenizer_image_token,tokenizer_multi_image_token
from PIL import Image

from einops import rearrange
from decord import VideoReader, cpu
import imageio
import cv2



@dataclass
class DataArguments:
    # data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path: List[str] = field(default_factory=list, metadata={"help": "List of paths to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: List[str] = field(default_factory=list, metadata={"help": "List of paths to the image folder."})
    # image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    max_images_num: int = 64
    is_multi_image: bool = False

    

def read_video(video_path, num_segments, bound=None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, num_segments, first_idx=0) 
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    # torch_imgs = self.transform(images_group)
    return images_group


def read_gif(video_path,num_segments, bound=None, fps=25):
    gif = imageio.get_reader(video_path)
    max_frame = len(gif) - 1
    
    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, num_segments, first_idx=0) 
    for index, frame in enumerate(gif):
        if index in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            img = Image.fromarray(img)
            images_group.append(img)
    # torch_imgs = self.transform(images_group)
    # return torch_imgs
    return images_group


def get_index(bound, fps, max_frame, num_segments, first_idx=0):
    """Uniformly sampled video frames

    Args:
        bound (list): _description_
        fps (_type_): _description_
        max_frame (_type_): _description_
        num_segments (_type_): _description_
        first_idx (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices





def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    print(
        f"WARNING: total file: {len(lst)}, random pick: {k}."
        f" (ignored)"
    )
    return new_lst


IMAGE_TOKEN_LENGTH = 768


def process_verse_input_ids(new_input_ids, expand_len, input_ids, end_idx, tokenizer):
    start_idx=0
    for i in range(end_idx, -1, -1):
        if expand_len < tokenizer.model_max_length:
            if input_ids[i] == IMAGE_TOKEN_INDEX:
                expand_len += IMAGE_TOKEN_LENGTH
                if expand_len > tokenizer.model_max_length:
                    start_idx=i
                    break
                else:
                    expand_len += 1
                    if input_ids[0] == tokenizer.bos_token_id and i == 1:
                        start_idx = i
                        break
                    elif not input_ids[0] == tokenizer.bos_token_id and i == 0:
                        start_idx = i
                        break
            else:
                expand_len += 1
                if input_ids[0] == tokenizer.bos_token_id and i == 1:
                    start_idx=i
                    break
                elif not input_ids[0] == tokenizer.bos_token_id and i == 0:
                    start_idx=i
                    break
        else:
            start_idx = i
            break

    new_input_ids.extend(input_ids[start_idx:end_idx+1])
    new_input_ids.extend([tokenizer.pad_token_id] * (tokenizer.model_max_length - len(new_input_ids)))
    return new_input_ids


def process_forward_input_ids(new_input_ids, expand_len, input_ids, start_idx, tokenizer):
    end_idx = 0
    for i in range(start_idx, len(input_ids)):
        if expand_len < tokenizer.model_max_length:
            if input_ids[i] == IMAGE_TOKEN_INDEX:
                expand_len += IMAGE_TOKEN_LENGTH
                if expand_len > tokenizer.model_max_length:
                    end_idx=i
                    break
                else:
                    expand_len += 1
                    if i == len(input_ids) - 1:
                        end_idx = i
                        break
            else:
                expand_len += 1
                if i == len(input_ids) - 1:
                    end_idx=i
                    break
        else:
            end_idx = i
            break

    new_input_ids.extend(input_ids[start_idx:end_idx])
    new_input_ids.extend([tokenizer.pad_token_id] * (tokenizer.model_max_length - len(new_input_ids)))
    return new_input_ids


def recursive_process_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer, index_list):
    if len(input_ids) - selected_index > selected_index:
        if input_ids[selected_index]==input_ids[selected_index+1]==input_ids[selected_index+2]:
            selected_index = random.choice(index_list)
            return recursive_process_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer, index_list)
        else:
            return process_forward_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer)
    else:
        if input_ids[selected_index]==input_ids[selected_index-1]==input_ids[selected_index-2]:
            selected_index = random.choice(index_list)
            return recursive_process_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer, index_list)
        else:
            return process_verse_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer)


def process_input_ids(input_ids, tokenizer):
    new_input_ids = []
    expand_len = 0

    if input_ids[0] == tokenizer.bos_token_id:
        new_input_ids.append(tokenizer.bos_token_id)
        expand_len += 1

    image_indices = [i for i, val in enumerate(input_ids) if val == IMAGE_TOKEN_INDEX]
    num_image_indices = len(image_indices)
    if num_image_indices<=1:
        return input_ids

    elif num_image_indices == 2:
        selected_index = image_indices[0]
        if len(input_ids) - selected_index > selected_index:
            return process_forward_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer)
        else:
            return process_verse_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer)
    else:
        selected_index=random.choice(image_indices)
        return recursive_process_input_ids(new_input_ids, expand_len, input_ids, selected_index, tokenizer,image_indices)


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    is_multi_image = data_args.is_multi_image
    max_images_num = data_args.max_images_num

    if not is_multimodal:
        return sources

    if is_multi_image:
        for i, source in enumerate(sources):
            if isinstance(source, str):
                sources[i] = source.replace("[ImageHere]", DEFAULT_IMAGE_TOKEN)
            else:
                for sentence in source:
                    image_token_num = sentence['value'].count("<image")
                    if image_token_num>max_images_num:
                        for i in range(max_images_num + 1, image_token_num + 1):
                            sentence['value'] = sentence['value'].replace(f"\n<image {i}>\n", " ")
    else:
        for i, source in enumerate(sources):
            if isinstance(source, str):
                sources[i] = source.replace("[ImageHere]", DEFAULT_IMAGE_TOKEN)
            else:
                for sentence in source:
                    if DEFAULT_IMAGE_TOKEN in sentence['value']:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                        sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                        sentence['value'] = sentence['value'].strip()

                    replace_token = DEFAULT_IMAGE_TOKEN

                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def minicpm_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_bunny_minicpm(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    is_multi_image: bool = False,
    has_image: bool = False,
    user_tokens=[1786, 4194, 95388],
    assistant_tokens=[1786, 10850, 95388],
) -> Dict:

    input_ids = [tokenizer.bos_token_id]
    targets = [IGNORE_INDEX]
    system_tokens=[1420, 12895, 2131, 1348, 22693, 3060, 1384, 1458, 17898, 17036, 16434, 72, 5, 2219, 16434, 6040, 11061, 95342,
     9450, 95342, 1384, 73252, 10864, 1385, 1358, 3060, 95361, 95328, 4483, 72]
    input_ids +=system_tokens
    targets += [IGNORE_INDEX] *len(system_tokens)
    # Apply prompt templates
    for i, source in enumerate(sources):
        for j, sentence in enumerate(source):
            role = sentence["from"]
            content = sentence["value"]
            if has_image:
                if role == "human":
                    content_ids = minicpm_tokenizer_image_token(content, tokenizer)
                    input_ids += user_tokens + content_ids + assistant_tokens
                    targets += [IGNORE_INDEX] * len(user_tokens) + [IGNORE_INDEX] * len(content_ids) \
                               + [IGNORE_INDEX] * len(assistant_tokens)

                elif role == "gpt":
                    content_ids = tokenizer.encode(content,add_special_tokens=False)
                    input_ids += content_ids
                    targets += content_ids
                    input_ids.append(tokenizer.eos_token_id)
                    targets.append(tokenizer.eos_token_id)

                else:
                    assert False
            else:
                if role == "human":
                    content_ids = tokenizer.encode(content)
                    input_ids += user_tokens + content_ids + assistant_tokens
                    targets += [IGNORE_INDEX] * len(user_tokens) + [IGNORE_INDEX] * len(content_ids)\
                               +[IGNORE_INDEX] * len(assistant_tokens)
                elif role == "gpt":
                    content_ids = tokenizer.encode(content)
                    input_ids += content_ids
                    targets += content_ids
                    input_ids.append(tokenizer.eos_token_id)
                    targets.append(tokenizer.eos_token_id)

                else:
                    assert False

    input_ids = [torch.LongTensor(input_ids)]
    targets = [torch.LongTensor(targets)]

    assert len(input_ids[0])==len(targets[0])
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        is_multi_image: bool = False,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and not is_multi_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image and is_multi_image:
        input_ids_list = []
        for prompt in conversations:
            if re.search(r"<image (\d+)>", prompt):
                input_ids_list.append(tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt'))
            else:
                input_ids_list.append(tokenizer_image_token(prompt, tokenizer, return_tensors='pt'))

        input_ids = torch.stack(input_ids_list, dim=0)

        # input_ids = torch.stack([tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and not is_multi_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))-1
            elif has_image and is_multi_image:
                if re.search(r"<image (\d+)>", conversation):
                    round_len = len(tokenizer_multi_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_multi_image_token(parts[0], tokenizer)) - 1
                else:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_bunny_gemma(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        is_multi_image: bool = False,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and not is_multi_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image and is_multi_image:
        input_ids = torch.stack([tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and not is_multi_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))-2
            elif has_image and is_multi_image:
                round_len = len(tokenizer_multi_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_multi_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)-2
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                assert False
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        is_multi_image: bool = False,
) -> Dict:

    conversations = []
    for source in sources:
        if isinstance(source, str):
            assert DEFAULT_IMAGE_TOKEN in source
            if is_multi_image:
                input_ids = [tokenizer_multi_image_token(source, tokenizer, return_tensors='pt')]
            else:
                input_ids = [tokenizer_image_token(source, tokenizer, return_tensors='pt')]
            for i in range(len(input_ids)):
                input_ids[i] = process_input_ids(input_ids[i], tokenizer)
                input_ids[i] = torch.tensor([tensor for tensor in input_ids[i]])

            targets = copy.deepcopy(input_ids)
            for target in targets:
                indices = torch.where(target == -200)
                target[indices] = IGNORE_INDEX
            return dict(input_ids=input_ids, labels=targets)

        assert len(source) == 2
        if is_multi_image:
            assert '\n<image' in source[0]['value']
            pattern = re.compile(r'<image (\d+)>')
            image_matches = pattern.findall(source[0]['value'])
            source[0]['value'] = ' '.join([f'<image {i}>' for i in image_matches])
        else:
            assert DEFAULT_IMAGE_TOKEN in source[0]['value']
            source[0]['value'] = DEFAULT_IMAGE_TOKEN

        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    if is_multi_image:
        input_ids = [tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if is_multi_image:
            tokenized_len = len(tokenizer_multi_image_token(source[0]['value'], tokenizer))
        else:
            tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def preprocess_bunny_with_bos(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        is_multi_image: bool = False,
        has_image: bool = False   
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())


    if has_image and not is_multi_image: # CWX BOS
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image and is_multi_image:
        input_ids = torch.stack(
            [tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        end_token_cnt = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and not is_multi_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_image and is_multi_image:
                round_len = len(tokenizer_multi_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_multi_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            end_token_cnt += 1
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        is_multi_image: bool = False,
        has_image: bool = False
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)

    if conversation_lib.default_conversation.version == "bunny":
        return preprocess_bunny(sources, tokenizer, is_multi_image,has_image=has_image)
    elif conversation_lib.default_conversation.version in {"minicpm", "llama", "phi3"}:
        return preprocess_bunny_with_bos(sources, tokenizer, is_multi_image,has_image=has_image)





class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.is_multi_image = data_args.is_multi_image
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        if self.is_multi_image:
            length_list = []
            for sample in self.list_data_dict:
                image_token_num = sample['conversations']['value'].count("<image")
                length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + 128*image_token_num)
            return length_list
        else:
            length_list = []
            for sample in self.list_data_dict:
                img_tokens = 128 if 'image' in sample else 0
                length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
            return length_list

    @property
    def modality_lengths(self):
        if self.is_multi_image:
            length_list = []
            for sample in self.list_data_dict:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                cur_len = cur_len if '<image' in sample else -cur_len
                length_list.append(cur_len)
            return length_list
        else:
            length_list = []
            for sample in self.list_data_dict:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                cur_len = cur_len if 'image' in sample else -cur_len
                length_list.append(cur_len)
            return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_file = image_file if isinstance(image_file, list) else [image_file]
            image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
            image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = [expand2square(i, tuple(int(x * 255) for x in processor.image_mean)) for i in image]
                image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            else:
                image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            self.is_multi_image,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class VideoLazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(VideoLazySupervisedDataset, self).__init__()
        self.is_multi_image = data_args.is_multi_image
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting input###########Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        if self.is_multi_image:
            length_list = []
            for sample in self.list_data_dict:
                image_token_num = sample['conversations']['value'].count("<image")
                length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + 128*image_token_num)
            return length_list
        else:
            length_list = []
            for sample in self.list_data_dict:
                img_tokens = 128 if 'image' in sample else 0
                length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
            return length_list

    @property
    def modality_lengths(self):
        if self.is_multi_image:
            length_list = []
            for sample in self.list_data_dict:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                cur_len = cur_len if '<image' in sample else -cur_len
                length_list.append(cur_len)
            return length_list
        else:
            length_list = []
            for sample in self.list_data_dict:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                cur_len = cur_len if 'image' in sample else -cur_len
                length_list.append(cur_len)
            return length_list


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
   
        if 'image' in sources[0]:
            
            image_file=self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            try:
                if ".gif" in  image_file:
                    videos= read_gif(video_path=os.path.join(image_folder, image_file),num_segments=8)
                else:
                    videos= read_video(video_path=os.path.join(image_folder, image_file),num_segments=8)

                image = processor.preprocess(videos, return_tensors='pt')   # 3, 8, 384, 384
            except:
                image = torch.zeros(3, 8, 384, 384)
   
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
     
        # image tensor [3*384*384] -> [3*4*384*384]
        
        image= list(torch.unbind(image, dim=1))
  
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            self.is_multi_image,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict





class ImageVideoLazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(ImageVideoLazySupervisedDataset, self).__init__()
        self.is_multi_image = data_args.is_multi_image
        # list_data_dict = json.load(open(data_path, "r"))

        list_data_dict=[]
        for path in data_path:
            if "itid" in path:
                files = glob.glob(path+"/*")
                list_data_dict+=files
            else:
                list_data_dict+=json.load(open(path, "r"))

        print("Formatting input###########Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        if self.is_multi_image:
            length_list = []
            for sample in self.list_data_dict:
                image_token_num = sample['conversations']['value'].count("<image")
                length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + 128*image_token_num)
            return length_list
        else:
            length_list = []
            for sample in self.list_data_dict:
                img_tokens = 128 if 'image' in sample else 0
                length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
            return length_list

    @property
    def modality_lengths(self):
        if self.is_multi_image:
            length_list = []
            for sample in self.list_data_dict:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                cur_len = cur_len if '<image' in sample else -cur_len
                length_list.append(cur_len)
            return length_list
        else:
            length_list = []
            for sample in self.list_data_dict:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                cur_len = cur_len if 'image' in sample else -cur_len
                length_list.append(cur_len)
            return length_list


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if ".json" in sources:  #itid
            data = json.load(open(sources, "r"))
            image=[]
            for index, base64_string in enumerate(data['base64_list']):
                image_data = base64.b64decode(base64_string)
                image_temp = Image.open(BytesIO(image_data))
                image_temp = image_temp.convert("RGB")
                image.append(image_temp)
            processor = self.data_args.image_processor
            image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            assert len(image)==len(data['base64_list'])

            source = ''.join(data['txt_list'])
            assert source.count("[ImageHere]")<=len(image)
            sources = preprocess_multimodal([copy.deepcopy(source)], self.data_args)
            data_dict = preprocess(
                sources,
                self.tokenizer,
                self.is_multi_image,
                has_image=(len(data['base64_list']) > 0))

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

            # image exist in the data
            if len(data['base64_list']) > 0:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            return data_dict

        elif "video" in self.list_data_dict[i]:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
    
            if 'image' in sources[0]:
                
                image_file=self.list_data_dict[i]["image"]
                image_folder = self.data_args.image_folder
                processor = self.data_args.video_processor
                
                for folder in image_folder:
                    
                    video_paths = os.path.join(folder, image_file)
                    if os.path.exists(video_paths):
                        try:
                            if ".gif" in  image_file:
                                videos = read_gif(video_path=video_paths, num_segments=64)
                            else:
                                videos = read_video(video_path=video_paths, num_segments=64) # 64 images list
                            image = processor.preprocess(videos, return_tensors='pt')   # 3, 64, 384, 384
                        except:
                            image = torch.zeros(3, 8, 384, 384)
                            print("Error reading file",video_paths)
                        break
    
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            
            image= list(torch.unbind(image, dim=1))
            data_dict = preprocess(
                sources,
                self.tokenizer,
                self.is_multi_image,
                has_image=('image' in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])

            # for video moment retrieval
            windows = sources[0][1]['value']
            winlst = ast.literal_eval(windows)
            data_dict['windows'] = winlst

            # image exist in the data
            if 'image' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            return data_dict

        else:
            if isinstance(i, int):
                    sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            if 'image' in sources[0]:
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                processor = self.data_args.image_processor
                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                for folder in image_folder:
                    image_paths = [os.path.join(folder, file) for file in image_file]
                    if all(os.path.exists(path) for path in image_paths):
                        image = [Image.open(path).convert('RGB') for path in image_paths]
                        break

                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = [expand2square(i, tuple(int(x * 255) for x in processor.image_mean)) for i in image]
                    image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess(
                sources,
                self.tokenizer,
                self.is_multi_image,
                has_image=('image' in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

            # image exist in the data
            if 'image' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        
        input_ids, labels, windows = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "windows"))
        windows = windows[0]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        if conversation_lib.default_conversation.version == "minicpm":
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.eos_token_id)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            attention_mask = input_ids.ne(self.tokenizer.eos_token_id)
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            windows=windows
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images
            batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ImageVideoLazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    # train_dataset = VideoLazySupervisedDataset(tokenizer=tokenizer,
    #                                       data_path=data_args.data_path,
    #                                       data_args=data_args)
    # train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                                       data_path=data_args.data_path,
    #                                       data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


