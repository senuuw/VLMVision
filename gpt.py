import torch, auto_gptq
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

text = '<ImageHere>Please describe this image in detail.'
image = 'test1.png'
with torch.cuda.amp.autocast():
    response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
print(response)
# The image features a quote by Oscar Wilde, "Live life with no excuses, travel with no regrets."
# The quote is displayed in white text against a dark background. In the foreground, there are two silhouettes of people standing on a hill at sunset.
# They appear to be hiking or climbing, as one of them is holding a walking stick.
# The sky behind them is painted with hues of orange and purple, creating a beautiful contrast with the dark figures.
