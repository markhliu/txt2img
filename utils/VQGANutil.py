

import sys
sys.path.append("./taming-transformers")


import requests, os

url1=('https://heibox.uni-heidelberg.de/f/'
      '140747ba53464f49b476/?dl=1')
url2=('https://heibox.uni-heidelberg.de/f/'
      '6ecf2af6c658432c8298/?dl=1')

file1="files/vqgan_imagenet_f16_1024.ckpt"
file2="files/vqgan_imagenet_f16_1024.yaml"    #A

if not os.path.exists(file1):
    fb=requests.get(url1)
    with open(file1,"wb") as f:
        f.write(fb.content)    #B

if not os.path.exists(file2):
    fb=requests.get(url2)
    with open(file2,"wb") as f:
        f.write(fb.content)    #C


import torch
torch.set_grad_enabled(False)    #A  

device = "cuda" if torch.cuda.is_available() else "cpu"

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def load_model():
    config = OmegaConf.load(file2)    #B
    model = VQModel(**config.model.params).to(device)    #C
    sd = torch.load(file1)["state_dict"]
    missing, unexpected = model.load_state_dict(sd,
                            strict=False)    #D
    return model



import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def process_image(file_name):    #A
    size=384
    img=PIL.Image.open(file_name)
    s = min(img.size)
    r = size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [size])
    return torch.unsqueeze(T.ToTensor()(img), 0)*2-1

def image_to_sequence(model,image):
    z=model.encoder(image.to(device))
    z=model.quant_conv(z) 
    z_q, _ , indices= model.quantize(z)    
    int_sequence = indices[2]    
    return int_sequence

def sequence_to_image(model,sequence):
    z_q2 = model.quantize.embedding(sequence)
    z_q2 = z_q2.permute(1,0).view((1,256,24,24))    
    z_post2 = model.post_quant_conv(z_q2)
    z_recon2 = model.decoder(z_post2)
    return z_recon2


