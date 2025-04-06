import os

MODEL_PATH = r"G:\FlashDnaProject\Llama-2-7B-Chat-GPTQ"
print(f"Checking if model path exists: {os.path.exists(MODEL_PATH)}")
print(f"Listing contents: {os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'Path not found'}")
print(os.path.realpath(MODEL_PATH))

