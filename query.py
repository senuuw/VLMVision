import os
import torch, auto_gptq
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM
from helper import detect_blur_and_bright_spot

auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)
if __name__ == '__main__':
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
def process_imagenx(image_path, model, tokenizer, prompt_list):
    response_list = []

    # autocast() to ensure that all torch objects on GPU
    with torch.cuda.amp.autocast():

        # Choose prompt
        for current_prompt in prompt_list:

            # Generate response and add to response list
            response = query(image_path, model, tokenizer, current_prompt)
            response_list.append(response)

    return response_list[0], response_list[1], response_list[2], response_list[3]


def query(image_path, model, tokenizer, prompt):
    # do_sample=False for shorter responses
    response, _ = model.chat(tokenizer, query=prompt, image=image_path, history=[], do_sample=False)
    return response

def process_directory(directory, model, tokenizer, prompt_list):
    # Dictionary that is turned into dataframe
    classification_dict = {
        'Filename': [],
        'Setting': [],
        'Lighting': [],
        'People': [],
        'Screens': [],
        'Blur': [],
        'Bright Spot': []
    }
    files_in_order = sorted(os.listdir(directory))
    print(files_in_order)
    for file_name in files_in_order:
        image_path = os.path.join(directory, file_name)

        # Query VLM for setting, lighting, motion and use OpenCV detection for blur and bright spot
        setting, lighting, people, screens = process_imagenx(image_path, model, tokenizer, prompt_list)
        blur, bright_spot = detect_blur_and_bright_spot(image_path)
        keys = list(classification_dict.keys())

        # Add to dictionary
        for i, classification in enumerate([file_name, setting, lighting, people, screens, blur, bright_spot]):
            classification_dict[keys[i]].append(classification)
        print(f"{file_name} done")

    # Convert to DataFrame with pandas
    data_frame = pd.DataFrame.from_dict(classification_dict)
    return data_frame

#directory_path = '52scenes'
scene_prompt = '<ImageHere> Please using only one word describe if the scene is outdoor or indoor.'
lighting_prompt = '<ImageHere> Please using only one word describe if the lighting in the image is bad or good.'
people_prompt = '<ImageHere> Please using only one word reply with True or False if there are people or body parts present.'
screen_prompt = ('<ImageHere> Please using only one word reply with True or False '
                 'if there are any television/computer/phone screens on present.')
prompt_list = [scene_prompt, lighting_prompt, people_prompt, screen_prompt]
#script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
#data_frame = process_directory(directory_path, model, tokenizer, prompt_list)
#data_frame.to_pickle('results2.pkl')

def process_frame_directory(frame_directory, model, tokenizer, prompt_list, results_directory):
    video_frame_folders = os.listdir(frame_directory)[5:]
    for video_folder in video_frame_folders:
        folder_path = os.path.join(frame_directory, video_folder)
        video_dataframe = process_directory(folder_path, model, tokenizer, prompt_list)
        pickle_output = os.path.join(results_directory, frame_directory)
        video_dataframe.to_pickle(pickle_output + '.pkl')
        print(f'{video_folder} Done')

process_directory("/home/sebastian/VLMVision/ego4d/frames/77cc4654-4eec-44c6-af05-dbdf71f9a401", model , tokenizer, prompt_list)