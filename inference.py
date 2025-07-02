import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
print("Loaded base model successfully.")
pipe.load_lora_weights("/home/bilal/FLUX.1-Kontext-dev-Training/saved_weights", weight_name="pytorch_lora_weights.safetensors")
print("Loaded LoRA weights successfully.")
pipe.to("cuda")
print("Pipeline moved to CUDA successfully.")
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
print("Input image loaded successfully.")
input_image = input_image.resize((1024, 1024))  # Resize the image to match the model's input size
image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5,
#   width=880,
#   height=1168
).images[0]

image.save("output.png")