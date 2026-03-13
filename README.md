# DC-ControlNet


# Content Encoder
You can use the pretrained ControlNet model based on SDXL. 

```bash
from pipelines.pipeline import StableDiffusionXLControlNetUnionPipeline
from models.controlnet_union import ControlNetModel_Union
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch
from diffusers.utils import load_image

controlnet = ControlNetModel_Union.from_pretrained("yang1232009/ControlNetPlus-SDXL").to(torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(torch.float16)

pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    "yang1232009/ControlNetPlus-SDXL",
    controlnet=controlnet,
    vae=vae,
    )
pipeline.to("cuda")
pipeline.to(torch.float16)

condition_image = load_image("./assets/condtion_images/guitar_normal.png")
condition_image = condition_image.resize((1024, 1024))

image = [0] * 8
image[3] = condition_image
# canny, hed, zoe, normal, sam, dot, box, mask
union_control_type = torch.Tensor([0,0,0,1,0,0,0,0])

prompt = 'A guitar'

positive_prompt = ", ultra highres, sharpness texture, High detail RAW Photo, shallow depth of field, dslr, film grain"
negative_prompt = "  blurry, disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w, cartoon, painting, illustration, worst quality, low quality"

generator = torch.Generator(device="cuda").manual_seed(42)
images = pipeline(
    prompt=prompt+positive_prompt, 
    negative_prompt=negative_prompt, 
    image_list=image, 
    union_control_type=union_control_type, 
    num_inference_steps=50, 
    generator=generator,
    num_images_per_prompt=1,
).images[0]

images.save("example.png")
```

