import argparse
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import decord
from decord import VideoReader, cpu
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,MULTI_IMAGE_TOKEN_1,MULTI_IMAGE_TOKEN_2,MULTI_IMAGE_TOKEN_3,MULTI_IMAGE_TOKEN_4,MULTI_IMAGE_TOKEN_5,MULTI_IMAGE_TOKEN_6,MULTI_IMAGE_TOKEN_7,MULTI_IMAGE_TOKEN_8
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import process_images,process_videos, tokenizer_image_token,tokenizer_multi_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
import os
from tqdm import tqdm
import json
from einops import rearrange
import time
import debugpy


def read_video(video_path,num_segments, bound=None):
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

def get_index(bound, fps, max_frame,num_segments, first_idx=0):
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

def get_duration(video_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    # 计算视频时长（以秒为单位）
    total_duration = (max_frame + 1) / fps
    return int(total_duration)


def video_loader_decord(root, second=0, end_second=None, chunk_len=-1, fps=-1, clip_length=32, jitter=False, return_tensors="pt"):
    if chunk_len == -1:
      
        vr = decord.VideoReader(root)
    
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        vid = None
        vr = decord.VideoReader(os.path.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)
    
    # load frames
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        if chunk_len==-1:
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()
        else:
            # find the remaining frames in the next chunk
            try:
                frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
                frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
                vid = None
                vr2 = decord.VideoReader(os.path.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
                frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
                frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
                frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
                frames = np.concatenate([frames_part1, frames_part2], axis=0)
            # the next chunk does not exist; the current chunk is the last one
            except (RuntimeError, decord.DECORDError) as error:
                print(error)
                frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
                frames = vr.get_batch(frame_ids).asnumpy()
            
    if return_tensors=='pt':
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
        return torch.stack(frames, dim=0)
    else:
        return frames


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def main(args):
    if args.debug:
        debugpy.listen(12345)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached")
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    
    tokenizer, model, image_processor, video_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.model_type)

    conv_mode = "bunny"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # video_file = "/zhaobai46d/panda/captioning/test_datasets/video_list/msvd_test.txt"
    # base_folder = "/zhaobai46d/panda/captioning"
    video_file = args.video_file
    base_folder = args.base_folder
    
    with open('/zhaobai46d/dataset/videodata/qvhighlights/highlight_val_release.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    content = []
    querys = []
    qid = []
    for item in data:
        qid.append(item['qid'])
        content.append(item['vid'] + '.mp4')
        querys.append(item['query'])



    # with open(video_file,"r") as f:
    #     content=f.readlines()
    
    result = []
    
    for k in tqdm(range(len(content))):
        i = content[k]
        i=i.strip()
        path=os.path.join(base_folder,i)
       
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles
        
        image=read_video(path, 64)
        
        # Similar operation in model_worker.py
        image_tensor = process_videos(image, video_processor, model.config)
        image_tensor = torch.unbind(image_tensor, dim=1)
        image_tensor=list(image_tensor)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=model.dtype)

        # inp="Write a title for this video."  # a man in a suit and tie is standing in front of a white board  
        inp = "According to the seconds of frame and video duration, find the relevant windows of the query."
        
        image_tokens = ""
        duration = get_duration(path)
        for i in range(1, 65, 1):
            second = int(duration / 64 * (i-1))
            image_tokens += f"{second}s <image {i}>\n"
        
        query = querys[k]
        # query = input("Please input query:\n")
        inp = 'Video: \n' + image_tokens + 'Video duration: ' + str(duration) + 's\nQuery: ' + query + '\n' + 'Task: ' + inp
        print(inp)
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
       
        input_ids = tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(outputs)
        
        result.append({'qid': qid[k], 'pred_relevant_windows': outputs.replace("<|endoftext|>","")})
    

    with open('data.jsonl', 'w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')
    # result = json.dumps(result, indent = 4)
    # with open("inference.json", "w") as f:
    #     f.write(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/zhaobai46d/videobunny/checkpoints-phi-2/bunny-lora-phi-2--8frame-2epoch-all")
    parser.add_argument("--model-base", type=str, default="/zhaobai46d/share/bunny/models/phi-2-msft")
    parser.add_argument("--model-type", type=str, default="phi-2")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="bunny")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base-folder", type=str, default="/zhaobai46d/dataset/videodata/qvhighlights/videos")
    parser.add_argument("--video-file", type=str, default="test_video.txt")
    args = parser.parse_args()
    main(args)
    
