import os
import shutil
import subprocess
from adaptive_sdk import Adaptive

# Define paths
src = "/mnt/fluidstack/nfs/llm-benchmarks"
dst = "fake_mount"

# Remove and recreate the fake mount directory
if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)

# Copy files from source to destination
shutil.copytree(src, dst, dirs_exist_ok=True)

# Set environment variables
os.environ["ADAPTIVE_ENDPOINT"] = "http://localhost:9000"
os.environ["HARMONY_ENDPOINT"] = "http://localhost:50053"
os.environ["ADAPTIVE_API_KEY"] = "key-CIXhEtlvl8UeAG7PEzdcHTG5DcYzhgxkfXQrY7wxbZH7exB02wQQIAuIWE3vWAMv"
os.environ["OUTPUT_DIR"] = dst

client = Adaptive("http://localhost:9000")

try:
    client.use_cases.create("test")
    client.models.attach("llama_3.1_8b_instruct", use_case="test")
except Exception as e:
    print("Use case or model already exists:", e)

try:
    client.use_cases.create("lora_test")
    client.models.attach("llama_3.1_8b_instruct", use_case="lora_test")
    client.models.attach("llama_3.1_sql_adapter", use_case="lora_test")
except Exception as e:
    print("Use case or model already exists:", e)

# import json
# for model in client.models.list():
#     print(model)
# # # # Run the Python script
subprocess.run(["python", "entrypoint.py"])
