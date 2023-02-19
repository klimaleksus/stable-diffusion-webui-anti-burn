**Latest update:** added `Stop` for breaking the first pass of highres.fix early; also fixed a bug with several samplers, which caused doubled steps tracking.

### Discussion: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/7464

# anti-burn

This is Extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for smoothing generated images by skipping a few very last steps and averaging together some images before them.

Also it can be used to save all intermediate steps as images.

## Installation:

This extension is included into the official index! Just use `Available → Load from` in WebUI Extensions tab.  
Or copy the link to this repository into `Install from URL`:
```
https://github.com/klimaleksus/stable-diffusion-webui-anti-burn
```
Also you may clone/download this repository and put it to `stable-diffusion-webui/extensions` directory.

## Usage:
You will see a section titled `Anti Burn (average smoothing of last steps images)` on txt2img and img2img tabs, which looks like this:

![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gui-fix.png)

Here is a copy of internal help documentation inside:
<details><summary>More info about Anti Burn</summary>

Sometimes samplers produce burned-out images on last step, especially on overtrained models at low number of steps.  
By simply **throwing off the last iteration** or two, you can get more appealing result!

Also, many samples create subtle differences on image in the end at large step count generation, but don't actually increase its quality: each iteration is slightly corrupted in finer details.  
By **averaging several last images**, you can get much smoother and accurate softer version!

This extension can do both: drop a few last images and merge together some of the rest.

Unless the first checkbox (`Enable Anti Burn`) is checked, this extension will be disabled (and its controls are grayed-out).  
Otherwise, it intercepts internal sampling loop call to grab latent results of each step and store a queue of them in RAM until the end of the batch. Then, those samples will be VAE-rendered and averaged together, replacing the final result for the rest of processing.

To help you select the right values for skip and count average, you can use `Brute-force mode`: this extension will loop through all possible combinations in chosen limits (taking into account actual number of available samples) and render them separated, into a subfolder inside your /txt2img-images/ (or what you have set) directory.

If you want to actually see all of intermediate steps images, you can check `Store`-mode: then this extension will render and save latents just as sampler produces them, so it is very slow (but accurate, since it is not dependent on "Live preview" settings). To see just a few last steps, you should rather use Brute mode with Count=0 but high Skip, it will be much faster, but you won't get samples from the first pass of highres-fix this way.

Third slider `Stop` can be used to abort the sampling process after the specified number of steps, counting from the start. This can help to drop some steps of the first pass of highres-fix, because it is not possible otherwise (since Count and Skip are effective only for the second pass if highres.fix was enabled); but you'll have to calculate your desired last step index manually.  
Also Stop might help you to make draft generations without highres-fix if you set your total steps high, but use Anti Burn to stop early; this way your result will be more stable (comparing to full-step version) than if you had asked for less steps without stopping.

_Note: earlier, there was a bug with latent grabbing for samplers that invoke the model twice per step (Heun, DPM2/a, DPM++ 2Sa, DPM++ SDE / Karras). For them, Anti Burn averaged half-steps instead of proper steps, giving less noticeable result. If you need to replicate that behavior, set the `Revert` checkbox._  
_Also, for adaptive (DPM fast and DPM adaptive) and compvis samples (DDIM and PMLS), the final image is slightly different than what was stored from last model call. Now the proper image is processed as extra step for those samplers (Revert disables this too) and used in averaging._

The `Debug`-mode will make this extension replace only a half of image with averaged version: it will redraw just regions of top-right and bottom-left corner. This might help you to understand, whether the averaging is really working, and simplifies comparing of different sources (for example, checking: is the image in Brute and the corresponding unaffected copy in Store are actually rendered properly in normal operation with the same Count and Skip values?)

Filename pattern for Brute:  
`AntiBurn_<start_timestamp>_Brute_<batch_number>_<image_number>-Skip=<now_skipped>-Average=<now_averaged>.png`  
Filename pattern for Store:  
`AntiBurn_<start_timestamp>_Store_<batch_number>_<image_number>[-Pass=<highres_pass>]-Step=<current_step>.png`

- This extension prints useful lines in console output and also stores `AntiBurn:` section to Generate info, but it doesn’t automatically read those parameters back.  
- When there are less total steps than selected Skip, then an original image is returned instead.  
- When there are less steps than needed for Count averaging, then it outputs "Average:X;" in generation info and proceeds with what is available.  
- When Count=1, no averaging is performed, so use can use it when you need just Skip. Since all modes (Store/Debug/Brute/Count/Skip) can be used together simultaneously, you can set Count=1 and Skip=0 if you want only checked Store to be in effect.  
- Be careful when using xformers: sometimes your GPU will create different images in a row, even with very same settings! So you won't be able to correctly replicate an image of some previous step, which might mislead you when you start comparing things.  
- Math for averaging: take float pixel colors by three channels for all needed samples; find a median (most common/mean value) for each pixel color between samples; then average all samples with equal-weight addition and division on count; finally mix together that median and average, scale to 0-255 and store as integers.  
- You cannot set Skip or Count just for the first pass of highres.fix pipeline (since internal array of stored latents must be cleared between passes). But now you may use Stop slider to set step number after which you want to abort the lowres pass and continue to second pass.
- When using Stop, console output may show one step less than requested, because aborting makes it to skip over updating progress bar.

**TL;DR**

If your image is ugly, try to set `Count`, about to 2-4.  
If your image is burned, try to increase `Skip`, about to 1-2, but set Count to 0.  
If you want really smooth result, set both Skip and Count to something **higher**.  
The more _generation Steps_ you have, the less AntiBurn effect you will get.

</details>

## Examples:

The following images are generated with 32 steps and CFG=7; the first image in each group is generated **without** using Anti Burn   
(Ctrl+Click on links to open them in new tabs to compare effectively)

### Model: `sd-v1-5-inpainting`

Average 4 steps (that's fine), sampler: `Euler a`

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_1_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_1_1.png)[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_1_2.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_1_2.png)

Average 16 steps (that's too much), sampler: `DDIM`

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_2_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_2_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_2_2.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_2_2.png)

Average 5 steps, skipping 3 steps using highres-fix at 0.7 denoising (with Lanczos and no [conditional mask fixes](https://github.com/klimaleksus/stable-diffusion-webui-conditioning-highres-fix)), sampler: `DPM adaptive` (it did 60 lowres steps + 42 highres steps)

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_3_1.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_3_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_3_2.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_3_2.png)

### Model: `protogenV22OfficialR_22`

Average 8 and skip 2 vs. average 12 skip 4, sampler: `DPM++ 2M Karras`
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_4_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_4_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_4_2.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_4_2.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_4_3.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_4_3.png)

Average 6 skipping 1, but with highres-fix at 0.5 denoising, sampler: `DPM++ SDE Karras` with 32 lowres steps and 12 highres steps (the WebUI setting _"With img2img, do exactly the amount of steps the slider specifies"_ was on):

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_5_1.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_5_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_5_2.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_5_2.png)

### Model: `dreamshaper252_252SafetensorFix`

Average 3, skip 3, sampler: `Heun`

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_6_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_6_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_6_2.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_6_2.png)

Just skip 1, but on 8 actual steps (instead of 32 to show the effect) with sampler `LMS`:

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_7_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_7_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_7_2.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_7_2.png)

Debug-mode demonstration: average all 32 steps, sampler `DPM fast`, notice blurred diagonal cells:

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_8_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_8_1.png)

### Model: `suzumehachi_V10`

Using Stop to affect the first pass of highres-fix: 0.6 denoising with Latent upscaler, sampler `Euler`, 24 lowres steps and 12 highres steps. No averaging but stop at 16:

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_1.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_1.png?)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_2.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_2.png?)

Corresponding before-highres-fix versions:

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_3.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_3.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_4.jpg?)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_9_4.png)

Using Stop to make a draft preview for generation by aborting at 16 steps, sampler `DPM++ 2S a`:

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_10_1.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_10_1.png)
[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_10_2.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_10_2.png)

This is how it would look like, if we'd asked for 16 steps in the first place:

[![](https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_10_3.jpg)]( https://klimaleksus2.ucoz.ru/sd/anti-burn/anti-burn_gallery-fix_10_3.png)

Since all samplers use different strategies for generation schedule, some are more appropriate to average heavily, while others will quickly become too blurry!

### EOF
