import os
import torch, auto_gptq
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM
from brightblur import detect_blur_and_bright_spot
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

def process_directory(directory, model, tokenizer, prompt_list, n):
    # Dictionary that is turned into dataframe
    classification_dict = {
        'Filename': [],
        'Setting': [],
        'Lighting': [],
        'Motion': [],
        'Blur': [],
        'Bright Spot': []
    }

    for file_name in os.listdir(directory):
        image_path = os.path.join(directory, file_name)

        # Query VLM for setting, lighting, motion and use OpenCV detection for blur and bright spot
        setting_list, lighting_list, motion_list = process_imagenx(image_path, model, tokenizer, prompt_list, n)
        blur, bright_spot = detect_blur_and_bright_spot(image_path)
        keys = list(classification_dict.keys())

        # Add to dictionary
        for i, classification in enumerate([file_name, setting_list, lighting_list, motion_list, blur, bright_spot]):
            classification_dict[keys[i]].append(classification)
        print(f"{file_name} done")

    # Convert to DataFrame with pandas
    data_frame = pd.DataFrame.from_dict(classification_dict)
    print(data_frame)
    return data_frame


def process_imagenx(image_path, model, tokenizer, prompt_list, n):
    response_list = []

    # autocast() to ensure that all torch objects on GPU
    with torch.cuda.amp.autocast():

        # Choose prompt
        for current_prompt in prompt_list:
            current_prompt_list = []

            # n is the amount of times each prompt is asked
            for i in range(n):
                response = query(image_path, model, tokenizer, current_prompt)
                current_prompt_list.append(response)

            response_list.append(current_prompt_list)
    return response_list[0], response_list[1], response_list[2]


def query(image_path, model, tokenizer, prompt):
    response, _ = model.chat(tokenizer, query=prompt, image=image_path, history=[], do_sample=False) #do_sample=False for shorter responses)
    return response


directory_path = 'frames'
scene_prompt = '<ImageHere> Please using only one word describe if the scene is outdoor or indoor.'
lighting_prompt = '<ImageHere> Please using only one word describe if the lighting in the image is bad or good.'
dynamic_prompt = '<ImageHere> Imagine we are trying to split images into categories where the objects and people are mostly static or dynamic, please answer with only one word static or dynamic for this image.'
script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
data_frame = process_directory(directory_path, model, tokenizer, [scene_prompt, lighting_prompt, dynamic_prompt], 3)
data_frame.to_pickle('results.pkl')


