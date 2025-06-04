import replicate
import os
import hashlib

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

def generate_image_url(prompt: str) -> str:
    try:
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": prompt,
                "width": 512,  # Explicit parameters help avoid version issues
                "height": 512
            }
        )
        return output[0] if output else get_fallback_image(prompt)
    except Exception as e:
        print(f"Image generation error: {e}")
        return get_fallback_image(prompt)

def get_fallback_image(prompt: str) -> str:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return f"https://dummyimage.com/600x400/000/fff&text={prompt[:30].replace(' ', '+')}+({prompt_hash[:6]})"