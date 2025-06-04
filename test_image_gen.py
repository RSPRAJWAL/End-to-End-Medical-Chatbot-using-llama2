from src.image_gen import generate_image_url
from dotenv import load_dotenv
import os

# Load environment variables first!
load_dotenv()  # ← Add this line

test_prompt = "Diagram of human heart anatomy"
result_url = generate_image_url(test_prompt)

print("\n" + "="*50)
print(f"Test Prompt: {test_prompt}")
print(f"Generated URL: {result_url}")
print("="*50 + "\n")