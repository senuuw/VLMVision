import os
import torch
import auto_gptq
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM
from query import process_directory
from frameextract import extract_video_frames
from helper import get_video_length
import time
import shutil
from multiprocessing import Process, set_start_method

auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]


def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    else:
        return torch.device("cpu")


def initialize_model(device):
    model = InternLMXComposer2QForCausalLM.from_quantized(
        'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True, device=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True)
    return model, tokenizer


def process_videos(video_list, frame_directory, results_directory, model, tokenizer, prompt_list, device):
    model.to(device)
    for video_name_mp4 in video_list:
        # Skip the specific problematic file
        if video_name_mp4 == "c3f5972e-9919-496c-a01a-75ffa5c7bcff.mp4":
            print(f"Skipping problematic video file: {video_name_mp4}")
            continue

        video_path = os.path.join(video_directory, video_name_mp4)
        pickle_file_name = f"{video_name_mp4[:-4]}.pkl"

        if pickle_file_name in completed_video_name_list:
            print(f"Video {video_name_mp4} already done, skipping")
            continue

        video_length = get_video_length(video_path)
        if video_length > 600:
            print(f"Video {video_name_mp4} more than 10 minutes, skipping")
            continue

        start = time.time()
        print(f"Starting {video_name_mp4} with length {video_length / 60:.2f} minutes")

        try:
            video_frame_directory = extract_video_frames(video_path, frame_directory, 1)
            video_dataframe = process_directory(video_frame_directory, model, tokenizer, prompt_list)
            pickle_output_path = os.path.join(results_directory, pickle_file_name)
            video_dataframe.to_pickle(pickle_output_path)
            print(
                f'Added {pickle_file_name} to results directory, completed in {(time.time() - start) / 3600:.2f} hours')
        except Exception as e:
            print(f"Error processing video {video_name_mp4}: {e}")
        finally:
            shutil.rmtree(video_frame_directory, ignore_errors=True)


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == '__main__':
    scene_prompt = '<ImageHere> Please using only one word describe if the scene is outdoor or indoor.'
    lighting_prompt = '<ImageHere> Please using only one word describe if the lighting in the image is bad or good.'
    people_prompt = '<ImageHere> Please using only one word reply with True or False if there are people or body parts present.'
    screen_prompt = ('<ImageHere> Please using only one word reply with True or False '
                     'if there are any television/computer/phone screens on present.')
    prompt_list = [scene_prompt, lighting_prompt, people_prompt, screen_prompt]
    # Set the start method to 'spawn'
    set_start_method('spawn')

    video_directory = "/home/sebastian/extssd/ego4d/v2/full_scale"
    frame_directory = "/home/sebastian/VLMVision/ego4d/frames"
    results_directory = "/home/sebastian/VLMVision/ego4d/results"

    # Get full list of video names, skip first two already done and manifest
    video_name_list = os.listdir(video_directory)[2:]
    completed_video_name_list = os.listdir(results_directory)

    # Split the video list between the two GPUs
    video_chunks = split_list(video_name_list, 2)

    processes = []

    for i, video_chunk in enumerate(video_chunks):
        device = get_device(i)
        model, tokenizer = initialize_model(device)

        p = Process(target=process_videos,
                    args=(video_chunk, frame_directory, results_directory, model, tokenizer, prompt_list, device))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
