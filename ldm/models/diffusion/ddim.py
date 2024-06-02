"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

dup_caption = None

def asy(x, a, b, c, d):
    return a - (a-b) * np.exp(-c*(x-d))  

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas) 
        self.register_buffer('ddim_alphas', ddim_alphas) 
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev) 
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               cd_extr,
               guidance_scale,
               base_count,
               prompt_fn,
               g3_count,
               urls,
               cond_fn=None,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        global dup_caption
        dup_caption = None
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates, simscore_list = self.ddim_sampling(conditioning,
                                                    size, 
                                                    cd_extr,
                                                    guidance_scale,
                                                    base_count,
                                                    prompt_fn,
                                                    g3_count,
                                                    urls,
                                                    cond_fn,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0, 
                                                    mask=mask, 
                                                    x0=x0, 
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature, 
                                                    score_corrector=score_corrector, 
                                                    corrector_kwargs=corrector_kwargs, 
                                                    x_T=x_T, 
                                                    log_every_t=log_every_t, 
                                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                                    unconditional_conditioning=unconditional_conditioning, 
                                                    )
        return samples, intermediates, simscore_list

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, 
                      cd_extr,
                      guidance_scale,
                      base_count,
                      prompt_fn,
                      g3_count,
                      urls,
                      cond_fn=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps 
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps) 
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0] 
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        simscore_list1 = []
        simscore_list2 = []
        simscore_list3 = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long) 

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img,
                                      cond, 
                                      ts, 
                                      cd_extr,
                                      guidance_scale,
                                      base_count,
                                      prompt_fn,
                                      g3_count,
                                      cond_fn,
                                      urls,
                                      index=index, 
                                      use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            if cond_fn is not None:
                img, pred_x0, simscore1, simscore2, simscore3, g3_count = outs
                simscore_list1.append(simscore1)
                simscore_list2.append(simscore2)
                simscore_list3.append(simscore3)
            else:
                img, pred_x0 = outs

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        if cond_fn is not None:
            fig = plt.figure()
            arr = np.arange(0,1001)
            guidance_schedule_3 = asy(arr,0.3,0.4,0.005,0)
            guidance_schedule_1 = asy(arr,0.3,0.4,0.005,0)
            arr2 = np.arange(1,1000,20)[::-1]
            plt.plot(arr2, np.array(simscore_list1), label='SSCD score before G2-3')
            plt.plot(arr2, np.array(simscore_list2), label='SSCD score after G2-3, before G1')
            plt.plot(arr2, np.array(simscore_list3), label='SSCD score after G1')
            plt.plot(arr, guidance_schedule_1, linestyle='dashed', label='thres_G1')
            plt.plot(arr, guidance_schedule_3, linestyle='dashed', label='thres_G3')
            plt.legend()
            fig.savefig(f"{base_count}_simscores.png")
            #torch.save(torch.from_numpy(np.array(simscore_list1)), f"{base_count}_simscore1.pt")
            #torch.save(torch.from_numpy(np.array(simscore_list2)), f"{base_count}_simscore2.pt")
            #torch.save(torch.from_numpy(np.array(simscore_list3)), f"{base_count}_simscore3.pt")
            return img, intermediates, simscore_list2
        else:
            return img, intermediates, None

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, cd_extr, guidance_scale, base_count, prompt_fn, g3_count, cond_fn, urls, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2) #shape [12,4,64,64]
            t_in = torch.cat([t] * 2) #shape [12]
            c_in = torch.cat([unconditional_conditioning, c]) #shape [12,77,768]
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2) #both shape [6,4,64,64]
            e_t_cond = e_t
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)
            e_t_backup = e_t

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device) #shape [6,1,1,1]
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        if cond_fn is not None:
            ### provide guidance ###
            arr = np.arange(0,1001)
            a=0.3
            b=0.4
            d=0
            guidance_schedule_3 = asy(arr,a,b,0.005,d)
            thres_3 = guidance_schedule_3[t[0]]
            guidance_schedule_1 = asy(arr,0.3,0.4,0.005,0)
            thres_1 = guidance_schedule_1[t[0]]

            if base_count > 20: 
                global dup_caption
                grads, simscore1, dcap  = cond_fn(x, e_t, cd_extr, guidance_scale, sqrt_one_minus_at, a_t, a_prev, t, base_count, urls)
                if dcap is not None:
                    dup_caption = dcap
                #print(f"cp1:{dup_caption}")
            else:
                simscore1 = 0

            simscore2 = simscore1
            simscore3 = simscore1

            if simscore1 > 0 and base_count > 20: 
                g2_scale = max(min(6.5, 6.5/0.5 * simscore1), 0) 
                if ((dup_caption is not None) and (simscore1 > thres_3)) or (g3_count != 10):
                    if t.item() > -1: 
                        if (dup_caption is not None) and (simscore1 > thres_3):
                            g3_count = 10
                
                        g3_count = g3_count - 1

                        if g3_count == 0:
                            g3_count = 10
                    
                        g3_scale = max(min(6.5 - g2_scale, (6.5 - g2_scale)/0.3 * simscore1), 0) 
                        prompts_nn = 1 * [dup_caption] 
                        c_nn = prompt_fn(prompts_nn)
                        x_in = torch.cat([x] * 3) 
                        t_in = torch.cat([t] * 3) 
                        c_in = torch.cat([unconditional_conditioning, c, c_nn]) 
                        e_t_uncond, e_t_prompt, e_t_caption = self.model.apply_model(x_in, t_in, c_in).chunk(3) 
                        e_t = e_t_uncond + (unconditional_guidance_scale - g2_scale) * (e_t_prompt - e_t_uncond) + g3_scale * (e_t_uncond - e_t_caption)       
                        grads, simscore2, dcap = cond_fn(x, e_t, cd_extr, guidance_scale, sqrt_one_minus_at, a_t, a_prev, t, base_count, urls)   
                        if dcap is not None:
                            dup_caption = dcap
                        #print(f"cp2:{dup_caption}")
                
                    simscore3 = simscore2
                    if simscore2 > thres_1: 
                        e_t = e_t - sqrt_one_minus_at * grads * (simscore2-0.3) * np.exp((1000-t.item())/170) #0.
                        #_, simscore3, dcap = cond_fn(x, e_t, cd_extr, guidance_scale, sqrt_one_minus_at, a_t, a_prev, t, base_count, urls)               
                        #if dcap is not None:
                        #    dup_caption = dcap
                        #print(f"cp3:{dup_caption}")
                else:
                    e_t = e_t_uncond + (unconditional_guidance_scale - g2_scale) * (e_t_cond - e_t_uncond)
                    grads, simscore2, dcap = cond_fn(x, e_t, cd_extr, guidance_scale, sqrt_one_minus_at, a_t, a_prev, t, base_count, urls)   
                    simscore3 = simscore2
                    if simscore2 > thres_1: #if simscore2 > 1:
                        e_t = e_t - sqrt_one_minus_at * grads * (simscore2-0.3) * np.exp((1000-t.item())/170)#0.
                        #_, simscore3, dcap = cond_fn(x, e_t, cd_extr, guidance_scale, sqrt_one_minus_at, a_t, a_prev, t, base_count, urls)  
                        #if dcap is not None:
                        #    dup_caption = dcap
                        #print(f"cp4:{dup_caption}")

        
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        if cond_fn is not None:
            return x_prev, pred_x0, simscore1, simscore2, simscore3, g3_count
        else:
            return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec