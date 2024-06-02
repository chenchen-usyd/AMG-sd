import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import requests
import json

import os
os.environ['CURL_CA_BUNDLE'] = ''

import clip
from clip_retrieval.clip_client import ClipClient, Modality
from PIL import Image as pimage
import urllib
import io
from torchvision import transforms
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

clip_model, clip_preprocess = clip.load("ViT-L/14", device="cpu", jit=True)
sscd = torch.jit.load("sscd_disc_mixup.torchscript.pt")

sscd_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

l2_transform = transforms.Compose([
    transforms.ToTensor(),
])

def L2(a, b):
    a = a
    b = b
    return torch.sqrt(torch.mean((a - b) ** 2, (1,2,3))) / 2

def l2_emb(img):
    img = sscd_transform(ToPILImage()(img[0]))
    img = img.unsqueeze(0)
    return img # shape [1, 3, 224, 224]

def sscd_emb(img):
    img = sscd_transform(ToPILImage()(img[0]))
    with torch.no_grad():
        emb = sscd(img.unsqueeze(0))[0,:]
    return emb # shape [512]

def sscd_emb_from_tensor(img):
    img_trans = F.interpolate(img.float(), size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        emb = sscd.to("cpu")(img_trans)[0,:]
    return emb 

def sscd_emb_from_file(img):
    img = sscd_transform(img)
    with torch.no_grad():
        emb = sscd(img.unsqueeze(0))[0,:]
    return emb 

def asy(x, a, b, c, d):
    return a - (a-b) * np.exp(-c*(x-d))   

client_1000 = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion5B-L-14",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=1000, 
    deduplicate=False, 
    use_safety_model=False,
    use_violence_detector=False
)

client = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion5B-L-14",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=1000, 
    deduplicate=True, 
    use_safety_model=False,
    use_violence_detector=False
)

def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        #headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    try:
        with urllib.request.urlopen(urllib_request, timeout=10) as r:
            img_stream = io.BytesIO(r.read())
        return img_stream
    except Exception as e:
        #print(f"Error loading image {url}: {e}")
        return None

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def get_text_emb(text):
    with torch.no_grad():
        text_emb = clip_model.encode_text(clip.tokenize([text], truncate=True).to("cpu"))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().detach().numpy().astype("float32")[0]
    return text_emb

def get_image_emb(image_url):
    with torch.no_grad():
        image = pimage.open(download_image(image_url))
        image_emb = clip_model.encode_image(clip_preprocess(image).unsqueeze(0).to("cpu")) #preprocess(image).unsqueeze(0).shape is [1,3,224,224]
        image_emb /= image_emb.norm(dim=-1, keepdim=True) # shape [1,768]
        image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
        return image_emb

def get_image_emb_from_tensor(image):
    with torch.no_grad():
        image = sscd_transform(ToPILImage()(image[0]))
        image_emb = clip_model.encode_image(image.unsqueeze(0).to("cpu")) 
        image_emb /= image_emb.norm(dim=-1, keepdim=True) 
        image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
        return image_emb # shape [1,768]

def get_image_emb_from_tran_tensor(image):
    with torch.no_grad():
        image_emb = clip_model.encode_image(image.unsqueeze(0).to("cpu")) 
        image_emb /= image_emb.norm(dim=-1, keepdim=True) 
        image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
        return image_emb # shape (768,)

def get_image_emb_from_file(image):
    with torch.no_grad():
        image = sscd_transform(image)
        image_emb = clip_model.encode_image(image.unsqueeze(0).to("cpu")) 
        image_emb /= image_emb.norm(dim=-1, keepdim=True) 
        image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
        return image_emb # shape (768,)

def open_and_transform(img_data, transform):
    try:
        return transform(pimage.open(img_data))
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml') 
model = load_model_from_config(config, f"models/ldm/stable-diffusion-v1/model.ckpt") 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# get learned conditioning of a given prompt
def prompt_fn(prompt):
    return model.get_learned_conditioning(prompt)

# get gradient for SSCD guidance
def cond_fn(x_t, e_t, cd_extr, guidance_scale, sqrt_one_minus_at, a_t, a_prev, t, base_count, urls):
    with torch.enable_grad():        
        cd_extr.eval()
        cd_extr = cd_extr.to("cuda:0")
        model.eval()
        x_t = x_t.detach().requires_grad_(True)
        
        # x_t -> x_0
        pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt() 
        
        # x_0 -> I0 decode first stage
        pred_I0 = model.differentiable_decode_first_stage(pred_x0) 
        pred_I0 = F.interpolate(pred_I0, size=(224, 224), mode='bilinear', align_corners=False) 
        I0_clip_emb = get_image_emb_from_tran_tensor(pred_I0[0])
        I0_sscd_emb = cd_extr(pred_I0)[0,:] 
        
        # Search kNN for I_0: find duplicated captions
        knn = client_1000.query(embedding_input=I0_clip_emb.tolist())
        #print(f"len(knn_1000): {len(knn)}")
        captions = []
        for item in knn:
            captions.append(item["caption"])

        count = Counter(captions)
        if count.most_common()[0][1] > 20:
            dup_caption = count.most_common()[0][0]
        else:
            dup_caption = None
        #print(f"Most commonly dup caption: {dup_caption} - {count.most_common()[0][1]} times")

        # Search kNN for I_0
        knn = client.query(embedding_input=I0_clip_emb.tolist())
        #print(f"len(knn_dedup): {len(knn)}")
        #print(knn)
        image_urls = set()
        for item in knn:
            image_urls.add(item["url"])
        image_urls = list(image_urls)
        
        # load user specified urls
        if len(urls) > 0:
            for url in urls:
                image_urls.append(url)
        
        #print(f"len(image_urls): {len(image_urls)}")
        # Use ThreadPoolExecutor to load images concurrently
        with ThreadPoolExecutor() as executor:
            images_data = list(executor.map(download_image, image_urls))
        images_data_list = []
        for img_data in images_data:
            if img_data is not None:
                try:
                    img_data_tran = sscd_transform(pimage.open(img_data))
                    if img_data_tran.shape[0] == 3:
                        images_data_list.append(img_data_tran)
                except Exception as e:
                    #print(f"Error opening image: {e}")
                    pass

        images_data = images_data_list
        del images_data_list
        images_data = torch.stack(images_data, dim=0).detach()
        #print(f"images_data.shape: {images_data.shape}")
        cd_extr = cd_extr.to("cpu")
        images_sscd_emb = cd_extr(images_data).detach()
        sscd_sims = torch.matmul(images_sscd_emb, I0_sscd_emb.cpu().detach()) 
        nn_index = torch.argmax(sscd_sims).detach()
        N0 = images_data[nn_index].unsqueeze(0).to("cuda:0").detach()
        cd_extr = cd_extr.to("cuda:0")
        N0_sscd_emb = cd_extr(N0)[0,:]
        N0_sscd_emb = N0_sscd_emb.detach()

        # Compute similarity score (SSCD)
        sim_score = torch.dot(I0_sscd_emb, N0_sscd_emb)
        sim_score.backward(retain_graph=True)
        grad = x_t.grad

        if False: # set to True if wish to view the intermediate results
            pred_I0_save = 255. * rearrange(pred_I0[0].detach().cpu().numpy(), 'c h w -> h w c')
            pred_I0_save = Image.fromarray(pred_I0_save.astype(np.uint8))
            pred_I0_save.save(f"{base_count}_{t[0]}_pred_I0_guided.png")

            N0_save = 255. * rearrange(N0[0].detach().cpu().numpy(), 'c h w -> h w c')
            N0_save = Image.fromarray(N0_save.astype(np.uint8))
            N0_save.save(f"{base_count}_{t[0]}_pred_N0_guided.png")

        return -grad * guidance_scale, sim_score.cpu().detach().numpy().item(), dup_caption

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--save_npz",
        action='store_true',
        help="save as npz file.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this many batches",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--urls",
        type=json.loads,
        help="urls of images to prevent memorization from",
        default=[]
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=100,
        help="scale of dissimilarity guidance",
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples 
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size 
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f: 
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    
    simscore_all_list = []
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"): 
                    for prompts in tqdm(data, desc="data"): 
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""]) 
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts) 
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f] 
                        samples_ddim, _, simscore_list = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c, 
                                                         batch_size=opt.n_samples,
                                                         shape=shape, 
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale, 
                                                         unconditional_conditioning=uc, 
                                                         eta=opt.ddim_eta, 
                                                         x_T=start_code,
                                                         cond_fn=cond_fn, 
                                                         g3_count=10,
                                                         urls = opt.urls,
                                                         prompt_fn=prompt_fn,
                                                         cd_extr=sscd,
                                                         guidance_scale=opt.guidance_scale, 
                                                         base_count=base_count
                                                         ) 

                        simscore_all_list.append(simscore_list)
                        
                        x_samples_ddim = model.decode_first_stage(samples_ddim) 
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy() 

                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        
                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if opt.save_npz or (not opt.skip_grid):
                            all_samples.append(x_checked_image_torch)

                if opt.save_npz:
                    all_samples_save = torch.stack(all_samples, 0)
                    all_samples_save = rearrange(all_samples_save.cpu().numpy(), 'n b c h w -> (n b) h w c')
                    save_path = "outputs/txt2img-samples/samples/gen_data.npz"
                    np.savez_compressed(save_path, data_x = all_samples_save, data_y = prompts[0]) 

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
