import os
from diffusers import StableDiffusionPipeline
import torch

device = "cuda"

# load model
model_path = "./output/" #pytorch_lora_weights.bin"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)

# load lora weights
pipe.unet.load_attn_procs(model_path)
# set to use GPU for inference
pipe.to(device)

# generate image
prompt = 'TEM data for gold multiply-twinned nanoparticle"'

prompt = prompt
out_dir = '40k_generated/{}'.format(prompt)
out_name = 'image'

if not os.path.exists(f'./{out_dir}'):
    os.makedirs(f'./{out_dir}')

#save image
for i in range(100):
    print(f"Generating image {i}")
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"./{out_dir}/{out_name}_{i}.jpg")
    
    
