import os
from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms

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
prompt1 = "GISAXS data with a beamstop in the center"


for i in range(100):
    image = pipe(prompt1, num_inference_steps=30).images[0]

    # process the generated image
    preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    im = preprocess(image)
    print(type(im))
    im = im.to(device)

    #load pre-trained discriminator
    DIS_PATH = './discriminator_data/checkpoints/resnet.pth'
    #discriminator = torch.jit.load(DIS_PATH)
    discriminator = torch.load(DIS_PATH)
    discriminator.eval()
    im = torch.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))

    outputs = discriminator(im)
    _, preds = torch.max(outputs, 1)

    #save image
    if preds:
        if not os.path.exists('./real'):
            os.makedirs('./real')
        
        image.save(f"./real/image1_{i}.png")
    else:
        if not os.path.exists('./fake'):
            os.makedirs('./fake')
        image.save(f"./fake/image1_{i}.png")

    
