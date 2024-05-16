import os
import torch, auto_gptq
import numpy as np
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM

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

def process_directory(directory, model, tokenizer, prompt_list, output_file, n):
    with open(output_file, 'w') as file:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            result_list = process_imagenx(image_path, model, tokenizer, prompt_list, n)
            file.write(f"Filename: {filename}\n")
            for i, result in enumerate(result_list):
                file.write(f"{result}\n")
            file.write('\n')
            print(f"Result for {filename}: {result_list}")

def process_imagenx(image_path, model, tokenizer, prompt_list, n):
    response_list= []
    with torch.cuda.amp.autocast():
        for current_prompt in prompt_list:
            current_prompt_list = []
            for i in range(n):
                response = query(image_path, model, tokenizer, current_prompt)
                current_prompt_list.append(response)
            response_list.append(current_prompt_list)
    return response_list


def query(image_path, model, tokenizer, prompt):
    response, _ = model.chat(tokenizer, query=prompt, image=image_path, history=[], do_sample=False) #do_sample=False for shorter responses)
    return response


directory_path = 'sampleframes'
scene_prompt = '<ImageHere> Please using only one word describe if the scene is outdoor or indoor.'
lighting_prompt = '<ImageHere> Please using only one word describe if the lighting in the image is bad or good.'
dynamic_prompt = '<ImageHere> Imagine we are trying to split images into categories where the objects and people are mostly static or dynamic. Please reply with only one word static or dynamic for this image'
script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
output_file = os.path.join(script_directory, 'sampleresults.txt')  # Output file in the same directory as the script
process_directory(directory_path, model, tokenizer, [scene_prompt, lighting_prompt, dynamic_prompt], output_file, 3)


