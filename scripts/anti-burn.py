# stable-diffusion-webui-anti-burn

'''
WebUI Dependencies:
    
1) Wrapped <modules.sd_samplers.create_sampler(*k,*kw)> - return value is used to get an object,
     that object is expected to possibly be <VanillaStableDiffusionSampler> or <KDiffusionSampler>.
2) If it has <.model_wrap_cfg> then it is considered to be KDiffusionSampler,
     then function is wrapped as <tensor=obj.model_wrap_cfg.forward(*k,**kw)>
3) If it has <.p_sample_ddim_hook> then it is considered to be VanillaStableDiffusionSampler,
     then function is wrapped as <tensor=(obj.p_sample_ddim_hook(*k,*kw))[1]>
4) <modules.processing.decode_first_stage(model,x)> is called, which does <model.decode_first_stage(x)>
5) <modules.processing.create_infotext(p,all_prompts,all_seeds,all_subseeds,comments=None,iteration=0,position_in_batch=0)>
     is called, expected to return a string info= for:
6) <modules.images.save_image(image,path,basename,seed,prompt,extension,info=None,p=None,forced_filename=None)>
'''

import time
import torch
import numpy as np
import gradio as gr
from PIL import Image
from modules.shared import opts
from modules import scripts, script_callbacks, processing, devices, sd_samplers
from modules import images as Images

class AntiBurnExtension(scripts.Script):
    def __init__(self,*k,**kw):
        self.latents = None 
        self.batch = 0
        self.phase = 0
        self.steps = 0
        self.block = 0
        super().__init__()
    def title(self):
        return 'Anti Burn (average smoothing of last steps images)'
    def show(self,is_img2img):
        return scripts.AlwaysVisible
    def ui(self, is_img2img):
        elem = 'stable-diffusion-webui-anti-burn_'+('img2img_' if is_img2img else 'txt2img_')
        with gr.Row(elem_id=elem+'row'):
            with gr.Accordion('Anti Burn (average smoothing of last steps images)',open=False,elem_id=elem+'accordion'):
                with gr.Row():
                    gr_enable = gr.Checkbox(label='Enable Anti Burn (and everything)',value=False,elem_id=elem+'enable')
                    gr_store = gr.Checkbox(label='Store all steps of inference also (slow!)',value=False,elem_id=elem+'store')
                with gr.Row():
                    gr_count = gr.Slider(minimum=1,maximum=64,step=1,label='Count of final steps to average together:',value=3,elem_id=elem+'count')
                with gr.Row():
                    gr_skip = gr.Slider(minimum=0,maximum=24,step=1,label='Skip this many very last steps: ',value=0,elem_id=elem+'skip')
                with gr.Row():
                    gr_debug = gr.Checkbox(label='Debug Anti Burn (output checked pattern 2×2 with averaged 2,3 cells overlay)',value=False,elem_id=elem+'debug')
                    gr_brute = gr.Checkbox(label='Brute-force mode (create Count×Skip separate images) ',value=False,elem_id=elem+'brute')
                with gr.Accordion('More info about Anti Burn',open=False,elem_id=elem+'help'):
                    gr.Markdown('''
Sometimes samplers produce burned-out images on last step, especially on overtrained models at low number of steps.  
By simply **throwing off the last iteration** or two, you can get more appealing result!

Also, many samples create subtle differences on image in the end at large step count generation, but don't actually increase its quality: each iteration is slightly corrupted in finer details.  
By **averaging several last images**, you can get much smoother and accurate softer version!

This extension can do both: drop a few last images and merge together some of the rest.

Unless the first checkbox (`Enable Anti Burn`) is checked, this extension will be disabled.  
Otherwise, it intercepts internal sampling loop call to grab latent results of each step and store a queue of them in RAM until the end of the batch. Then, those samples will be VAE-rendered and averaged together, replacing the final result for the rest of processing.

To help you select the right values for skip and count average, you can use `Brute-force mode`: then this extension will loop through all possible combinations in chosen limits (taking into account actual number of available samples) and render them separated, into a subfolder inside your /txt2img-images/ (or what you have set) directory.  

If you want to actually see all of intermediate steps images, you can check `Store`-mode: then this extension will render and save latents just as sampler produces them, so it is very slow (but accurate, since it is not dependent on "Live preview" settings). To see just a few last steps, you should rather use Brute mode with "Count=0" but high Skip, it will be much faster, but you won't get samples from the first pass of highres-fix this way.  

The `Debug`-mode will make this extension replace only a half of image with averaged version: it will redraw just regions of top-right and bottom-left corner. This might help you to understand, whether the averaging is really working, and simplifies comparing of different sources (for example, checking: is the image in Brute and the corresponding unaffected copy in Store are actually rendered properly in normal operation with the same Count and Skip values?)  

Filename pattern for Brute:  
`AntiBurn_<start_timestamp>_Brute_<batch_number>_<image_number>-Skip=<now_skipped>-Average=<now_averaged>.png`  
Filename pattern for Store:  
`AntiBurn_<start_timestamp>_Store_<batch_number>_<image_number>-Phase=<highres_pass>-Step=<current_step>.png`

- This extension prints useful lines in console output and also stores `AntiBurn:` section to Generate info, but it doesn’t automatically read those parameters back.  
- When there are less total steps than selected Skip, then an original image is returned instead.  
- When there are less steps than needed for Count averaging, then it outputs "Average:X;" in generation info and proceeds with what is available.  
- When Count=1, no averaging is performed, so use can use it when you need just Skip. Since all modes (Store/Debug/Brute/Count/Skip) can be used together simultaneously, you can set Count=1 and Skip=0 if you want only checked Store to be in effect.  
- Be careful when using xformers: sometimes your GPU will create different images in a row, even with very same settings! So you won't be able to correctly replicate an image of some previous step, which might mislead you when you start comparing things.  
- Math for averaging: take float pixel colors by three channels for all needed samples; find a median (most common/mean value) for each pixel color between samples; then average all samples with equal-weight addition and division on count; finally mix together that median and average, scale to 0-255 and store as integers.  
- You cannot set Skip or Count just for the first pass of highres.fix pipeline. Thus, if you use latent upscale, you're out of luck already; and otherwise you'll have to perform first lowres phase manually, if you really have a reason to do so.

**TL;DR**

If your image is ugly, try to set `Count`, about to 2-4.  
If your image is burned, try to increase `Skip`, about to 1-2, but set Count to 0.  
If you want really smooth result, set both Skip and Count to something **higher**.  
The more _generation Steps_ you have, the less AntiBurn effect you will get.
                    ''')
        return [gr_enable,gr_debug,gr_count,gr_skip,gr_brute,gr_store]

    def process(self,p,gr_enable,gr_debug,gr_count,gr_skip,gr_brute,gr_store):
        if not gr_enable:
            return
        self.block = 0
        self.start = int(time.time())
        if gr_store:
            print('AntiBurn: store all steps (slow!)')
    
    def postprocess(self,p,processed,*args,**kw):
        if hasattr(sd_samplers.create_sampler,'__anti_burn_wrapper'):
            orig = getattr(sd_samplers.create_sampler,'__anti_burn_wrapper')
            if orig is not None:
                sd_samplers.create_sampler = orig

    def process_batch(self,p,gr_enable,gr_debug,gr_count,gr_skip,gr_brute,gr_store,batch_number,prompts,seeds,subseeds,**kw):
        old = sd_samplers.create_sampler
        if hasattr(old,'__anti_burn_wrapper'):
            orig = getattr(old,'__anti_burn_wrapper')
            if orig is not None:
                old = orig
                if not gr_enable:
                    sd_samplers.create_sampler = old
                    return
        if not gr_enable:
            return;
        self.block = 1
        self.steps = 0
        self.phase = 0
        self.batch = batch_number+1
        if gr_skip>0 or gr_count>1 or gr_brute or not gr_store:
            self.latents = []
        else:
            self.latents = None

        def hook_sampler(res):
            if self.block<1:
                return
            length = gr_skip+gr_count

            def wrapped_grab(tensor):
                if self.block<2:
                    return
                self.steps += 1
                self.device = tensor.device
                latents = self.latents
                if latents is not None:
                    will = len(latents)+1
                    if will>1 and latents[0].numel()!=tensor.numel():
                        latents.clear()
                        will = 1
                        self.steps = 1
                        self.phase += 1
                    latents.append(tensor.cpu())
                    if will>length:
                        latents.pop(0)
                if gr_store:
                    batch = [(torch.clamp((processing.decode_first_stage(p.sd_model,torch.stack([batch.to(dtype=devices.dtype_vae,device=self.device)]))[0].to(device='cpu',dtype=torch.float32)+1.0)/2.0,min=0.0,max=1.0)).numpy() for batch in tensor]
                    batch = (np.moveaxis(batch,1,3)*255.0).astype(np.uint8)
                    p.extra_generation_params['AntiBurn'] = 'Store;Phase={};Step={}'.format(self.phase,self.steps)
                    iteration = self.batch-1 if self.batch>0 else 0
                    text = '-Phase={}-Step={}'.format(self.phase,self.steps)
                    save_to_dirs = opts.save_to_dirs
                    opts.save_to_dirs = False
                    i = 0
                    for img in batch:
                        path = 'AntiBurn_{}_Store_{}_{}'.format(self.start,self.batch,i+1)
                        try:
                            infotext = processing.create_infotext(p,p.all_prompts,p.all_seeds,p.all_subseeds,comments=[],iteration=iteration,position_in_batch=i)
                            Images.save_image(Image.fromarray(img),p.outpath_samples+'/'+path,p.all_seeds[i],p.all_prompts[i],opts.samples_format,info=infotext,p=p,forced_filename=path+text)
                        except:
                            traceback.print_exc()
                        i += 1
                    del p.extra_generation_params['AntiBurn']
                    opts.save_to_dirs = save_to_dirs
            
            def wrapped_cfg(*k,**kw):
                res = old_forward(*k,**kw)
                wrapped_grab(res)
                return res
            def wrapped_ddim(*k,**kw):
                res = old_forward(*k,**kw)
                wrapped_grab(res[1])
                return res
            
            old_forward = None
            self.phase += 1
            self.steps = 0
            if self.latents is not None:
                self.latents.clear()
            self.block = 2
            if hasattr(res,'model_wrap_cfg'):
                old_forward = res.model_wrap_cfg.forward
                setattr(res.model_wrap_cfg,'forward',wrapped_cfg)
            elif hasattr(res,'p_sample_ddim_hook'):
                old_forward = res.p_sample_ddim_hook
                setattr(res,'p_sample_ddim_hook',wrapped_ddim)
            else:
                print('AntiBurn: unknown sampler?')

        def my_create_sampler(*k,**kw):
            orig = getattr(my_create_sampler,'__anti_burn_wrapper')
            res = orig(*k,**kw)
            hook_sampler(res)
            return res
        
        sd_samplers.create_sampler = my_create_sampler
        setattr(my_create_sampler,'__anti_burn_wrapper',old)
        if hasattr(p,'sampler') and p.sampler is not None:
            hook_sampler(p.sampler)

    def postprocess_batch(self,p,gr_enable,gr_debug,gr_count,gr_skip,gr_brute,gr_store,images,batch_number,**kw):
        self.block = 0
        if not gr_enable or self.latents is None:
            return
        if gr_skip!=0 and not gr_brute:
            latents = self.latents[:-gr_skip]
        else:
            latents = self.latents
        self.latents = None
        count = len(latents)
        if gr_brute:
            need = []
            for skip in range(0,gr_skip+1):
                have = count-skip
                if have>0:
                    if have>gr_count:
                        have = gr_count
                    for j in range(have):
                        need.append((skip,j+1))
            total = len(need)
            if total==0:
                print('AntiBurn: brute - nothing to do')
                return
            over = False
            average = count-gr_skip
        else:
            average = count
            if average==0:
                print('AntiBurn: nothing to do')
                return
            over = True
            total = 1
            need = [(gr_skip,average)]
        if average>0:
            info = 'Count='+str(gr_count)+';'+('Skip='+str(gr_skip)+';' if gr_skip>0 else '')+('Debug;' if gr_debug else '')+('Average='+str(average)+';' if average!=gr_count else '')
            print('AntiBurn: '+info)
        else:
            print('AntiBurn: main task empty')
        arrs = []
        for latent in latents:
            batches = []
            for batch in latent:
                batch = batch.to(dtype=devices.dtype_vae,device=self.device)
                batch = torch.stack([batch])
                batch = processing.decode_first_stage(p.sd_model,batch)[0]
                batch = batch.to(device='cpu',dtype=torch.float32)
                batch = torch.clamp((batch+1.0)/2.0,min=0.0,max=1.0)
                batches.append(batch.numpy())
            arrs.append(batches)
        del latents
        step = 0
        for skip,have in need:
            step += 1
            arr = arrs
            if gr_brute:
                print('AntiBurn: brute {}/{}, Skip={}, Average={}'.format(step,total,skip,have))
                if skip>0:
                    arr = arrs[:-skip]
                diff = len(arr)-have
                if diff>0:
                    arr = arr[diff:]
            if have!=len(arr):
                print('AntiBurn: internal error?')
                have = len(arr)
            tgt = np.copy(arr[0])
            if have>1:
                for i in range(1,have):
                    tgt += arr[i]
                tgt /= have
                med = np.median(np.moveaxis(arr,0,4),axis=4,overwrite_input=over)
                tgt = (tgt+med)/2
                del med
            if gr_brute:
                batch = (np.moveaxis(tgt,1,3)*255.0).astype(np.uint8)
                p.extra_generation_params['AntiBurn'] = 'Brute;Skip={};Average={}'.format(skip,have)
                iteration = self.batch-1 if self.batch>0 else 0
                text = '-Skip={}-Average={}'.format(skip,have)
                save_to_dirs = opts.save_to_dirs
                opts.save_to_dirs = False
                i = 0
                for img in batch:
                    path = 'AntiBurn_{}_Brute_{}_{}'.format(self.start,self.batch,i+1)
                    try:
                        infotext = processing.create_infotext(p,p.all_prompts,p.all_seeds,p.all_subseeds,comments=[],iteration=iteration,position_in_batch=i)
                        Images.save_image(Image.fromarray(img),p.outpath_samples+'/'+path,p.all_seeds[i],p.all_prompts[i],opts.samples_format,info=infotext,p=p,forced_filename=path+text)
                    except:
                        traceback.print_exc()
                    i += 1
                del p.extra_generation_params['AntiBurn']
                opts.save_to_dirs = save_to_dirs
                del batch
        if gr_brute:
            if average<=0:
                return
        tgt = torch.from_numpy(tgt)
        if gr_debug:
            (b,c,w,h) = images.size()
            w2 = w//2
            h2 = h//2
            images[:,:,:w2,h2:] = tgt[:,:,:w2,h2:]
            images[:,:,w2:,:h2] = tgt[:,:,w2:,:h2]
        else:
            images[:] = tgt[:]
        p.extra_generation_params['AntiBurn'] = info
        devices.torch_gc()

def AntiBurnExtension_unloaded():
    if hasattr(sd_samplers.create_sampler,'__anti_burn_wrapper'):
        orig = getattr(sd_samplers.create_sampler,'__anti_burn_wrapper')
        if orig is not None:
            sd_samplers.create_sampler = orig
script_callbacks.on_script_unloaded(AntiBurnExtension_unloaded)

#EOF
