import os
import cv2
import torch
import numpy as np
from PIL import Image
from itertools import islice

from ldm.util import instantiate_from_config

import torch.nn.functional as F
from torchvision.transforms import ToPILImage

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
    return emb # shape [512]

def sscd_emb_from_file(img):
    img = sscd_transform(img)
    with torch.no_grad():
        emb = sscd(img.unsqueeze(0))[0,:]
    return emb # shape [512]

def asy(x, a, b, c, d):
    return a - (a-b) * np.exp(-c*(x-d))   

client = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion5B-L-14",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=1000, # change this later
    deduplicate=True,
    use_safety_model=True,
    use_violence_detector=True
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

# Define the function to safely open the image and apply the transform
def open_and_transform(img_data, transform):
    try:
        # Try to open the image and apply the transform
        return transform(pimage.open(img_data))
    except Exception as e:
        # If an error occurs, print the error message and return None
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

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

sscd_sim_list = []
sscd_sim_list_anchor = []
clip_score_list = []

sample_path_eval = "outputs/txt2img-samples/samples_nn"
if not os.path.isdir(sample_path_eval):
    os.mkdir(sample_path_eval)
base_count = len(os.listdir(sample_path_eval))

save_path = "outputs/txt2img-samples/samples/gen_data.npz"
loaded = np.load(save_path)
data_x = loaded['data_x']
data_y = loaded['data_y']

for item in data_x:
    x_checked_image_torch = torch.from_numpy(item).permute(2, 0, 1).unsqueeze(0)
    x_clip_emb = get_image_emb_from_tensor(x_checked_image_torch) 
    x_sscd_emb = sscd_emb_from_tensor(x_checked_image_torch) 

    knn = client.query(embedding_input=x_clip_emb.tolist())
    image_urls = set()
    for item in knn:
        image_urls.add(item["url"])
    image_urls = list(image_urls)
                        
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
    images_sscd_emb = sscd(images_data).detach()
    sscd_sims = torch.matmul(images_sscd_emb, x_sscd_emb.cpu().detach())
    nn_index = torch.argmax(sscd_sims).detach()
    nn_image = images_data[nn_index]
    nn_sscd = sscd_sims[nn_index]
    nn_emb = images_sscd_emb[nn_index]

    sscd_sim_list.append(nn_sscd)
    y_clip_emb = get_text_emb(data_y.item()) 
    clip_score = np.dot(x_clip_emb, y_clip_emb) * 100
    clip_score_list.append(clip_score)

    print("OK.")    

print("sscd_sim_list: {}".format(sscd_sim_list))
print("clip_score_list: {}".format(clip_score_list))

print("Saved to ***.pt files")
print("clip_score: {}".format(clip_score_list))
print("sscd_sim: {}".format(sscd_sim_list))
print("{} out of {} generations have SSCD > 0.5".format(np.sum(np.array(sscd_sim_list) > 0.5), len(sscd_sim_list)))
print("{} out of {} generations have SSCD > 0.4".format(np.sum(np.array(sscd_sim_list) > 0.4), len(sscd_sim_list)))
print("mean CLIP score is: {:.2f}".format(np.mean(clip_score)))
print("SSCD Top 5%: {:.4f}".format(np.quantile(sscd_sim_list, 0.95)))
print("SSCD Top 1: {:.4f}".format(np.max(sscd_sim_list)))
print("finish")
