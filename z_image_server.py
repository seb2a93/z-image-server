from diffusers import ZImagePipeline
import torch
from flask import Flask, request, send_file
import io
import random

app = Flask(__name__)

print("⏳ Chargement Z-Image Turbo...")
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
print("✅ Prêt!")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', 'a cat')
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    seed = data.get('seed', random.randint(1, 2**32-1))
    
    print(f"🎨 Génération: {prompt[:50]}... ({width}x{height}, seed={seed})")
    
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    
    print("✅ Image générée!")
    return send_file(img_io, mimetype='image/png')

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
