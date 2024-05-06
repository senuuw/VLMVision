import os
import torch, auto_gptq
from PIL import Image
import torchvision.transforms as transforms
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

def process_image(image_path, model, tokenizer, prompt):
    with torch.cuda.amp.autocast():
        response, _ = model.chat(tokenizer, query=prompt, image=image_path, history=[], do_sample=False) #do_sample=False for shorter responses
    return response
def process_directory(directory, model, tokenizer, prompt, output_file):
    with open(output_file, 'w') as file:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            result = process_image(image_path, model, tokenizer, prompt)
            file.write(f"Filename: {filename}\nResult: {result}\n\n")
            print(f"Result for {filename}: {result}")

# Usage example
directory_path = 'examples'
prompt = '<ImageHere> Please describe this image in detail.'
script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
output_file = os.path.join(script_directory, 'results.txt')  # Output file in the same directory as the script
process_directory(directory_path, model, tokenizer, prompt, output_file)
