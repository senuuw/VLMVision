import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM
import os
from query import process_frame_directory
from frameextract import extract_video_frames
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

scene_prompt = '<ImageHere> Please using only one word describe if the scene is outdoor or indoor.'
lighting_prompt = '<ImageHere> Please using only one word describe if the lighting in the image is bad or good.'
people_prompt = '<ImageHere> Please using only one word reply with True or False if there are people or body parts present.'
screen_prompt = ('<ImageHere> Please using only one word reply with True or False '
                 'if there are any television/computer/phone screens on present.')
prompt_list = [scene_prompt, lighting_prompt, people_prompt, screen_prompt]

ego4d_path = '/home/sebastian/extssd/ego4d/v2/full_scale'
def main(ego4_path):
    video_list = os.listdir(ego4_path)[1:10]
    for video_name in video_list:
        video_path = os.path.join(ego4_path, video_name)
        print(video_path)