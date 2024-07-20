import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, tokenizer_multi_image_token,get_model_name_from_path,process_images
from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN,MAX_IMAGE_LENGTH
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type)

    # test_prompt='I will send you two screenshots of the book cover. Please examine the screenshots and answer a question. You must choose your answer from the Choice List.   \n<image 1>\nDay of the Dead Activity Book \n<image 2>\nNutrition and Disease Management for Veterinary Technicians and Nurses \n<image 3>\nI Regret Everything: A Love StoryQuestion: Please give the correct text order corresponding to each bookchoose choice_list: 0,1,2, 0,2,1, 1,0,2, 1,2,0, 2,0,1, 2,1,0'
    test_prompt='Based on the stories accompanying the first images, can you devise a conclusion for the story that incorporates the last image? \n<image 1>\nCaption#1:the woman starts to speak.\n<image 2>\nCaption#2:the woman begins to speak.\n<image 3>\nCaption#3:'
    input_ids = tokenizer_multi_image_token(test_prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
    print(input_ids)
    image_file=['/zhaobai46d/share/data/I4-Core/DiDeMoSV/core/images/33.jpg','/zhaobai46d/share/data/I4-Core/DiDeMoSV/core/images/34.jpg',
                '/zhaobai46d/share/data/I4-Core/DiDeMoSV/core/images/35.jpg']

    image_file = image_file if isinstance(image_file, list) else [image_file]
    image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
    image = [Image.open(file).convert('RGB') for file in image_file]
    device = next(model.parameters()).device
    image_tensor = process_images(image, image_processor, model.config).half().to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True,)
    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         # images=image_tensor,
    #         do_sample=True if args.temperature > 0 else False,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #         num_beams=args.num_beams,
    #         # no_repeat_ngram_size=3,
    #         max_new_tokens=1024,
    #         use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    print('output',outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
