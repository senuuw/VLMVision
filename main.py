import os
import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM
from query import process_directory
from frameextract import extract_video_frames
import time

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


# init model and tokenizer
model = InternLMXComposer2QForCausalLM.from_quantized(
    'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True, device="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(
    'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True)

text = '<ImageHere>Please describe this image in detail.'

scene_prompt = '<ImageHere> Please using only one word describe if the scene is outdoor or indoor.'
lighting_prompt = '<ImageHere> Please using only one word describe if the lighting in the image is bad or good.'
people_prompt = '<ImageHere> Please using only one word reply with True or False if there are people or body parts present.'
screen_prompt = ('<ImageHere> Please using only one word reply with True or False '
                 'if there are any television/computer/phone screens on present.')

prompt_list = [scene_prompt, lighting_prompt, people_prompt, screen_prompt]


def main(video_directory, frame_directory, results_directory, model, tokenizer, prompt_list):
    #Get full list of video names
    video_name_list = os.listdir(video_directory)[1:]

    #Create or use existing frame and results directories
    os.mkdir('/home/sebastian/VLMVision/ego4d')
    if os.path.exists(frame_directory):
        print('Frame directory exists already')
    else:
        os.mkdir(frame_directory)
        print(f'Creating new frame directory at {frame_directory}')

    if os.path.exists(results_directory):
        print('Results directory exists already')
    else:
        os.mkdir(results_directory)
        print(f'Creating new results directory at {results_directory}')

    # Get full list of videos with results alredy
    completed_video_name_list = os.listdir(results_directory)

    video_count = 0

    for video_name in video_name_list:
        if f"{video_name}.pkl" in completed_video_name_list:
            print(f"Video {video_name} already completed")
        else:
            start = time.time()
            video_path = os.path.join(video_directory, video_name)
            # Extract 1 frame per second to frame_directory and return video frame directory path
            # Will print "Extracted {frame_count} frames to {video_frame_directory}"
            video_frame_directory = extract_video_frames(video_path, frame_directory, 1)

            # Process and create dataframe for video frames, return dataframe
            # Prints done for every frame (to be removed after testing)
            video_dataframe = process_directory(video_frame_directory, model, tokenizer, prompt_list)

            #Save dataframe in pickle
            pickle_output_path = os.path.join(results_directory, video_name, '.pkl')
            video_dataframe.to_pickle(pickle_output_path)
            video_count += 1
            end = time.time()
            print(f'Added {video_name} to results directory, Video #{video_count} completed in {end - start:.2f}')
            os.remove(video_frame_directory)

video_directory = "/home/sebastian/extssd/ego4d/v2/full_scale"
frame_directory = "/home/sebastian/VLMVision/ego4d/frames"
results_directory = "/home/sebastian/VLMVision/ego4d/results"
main(video_directory, frame_directory, results_directory, model, tokenizer, prompt_list)