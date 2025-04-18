import os
from pathlib import Path
import subprocess
import datetime
from typing import Literal
import requests
import json


OUTPUT_DIR = "./llm-benchmarks"

# --- Script Setup ---

GPU_NAME = os.environ.get("GPU_NAME", "")

BENCHMARKS_PAGE = "https://inference-benchmarks.tech-adaptive-ml.com/reports"
HARMONY_ENDPOINT = "http://adaptive-harmony-0.adaptive-harmony-hdls-svc.default.svc.cluster.local:50053"

# Get Adaptive version from Harmony
try:
    resp = requests.get(f"{HARMONY_ENDPOINT}/version_info", timeout=5)
    resp.raise_for_status()
    ADAPTIVE_VERSION = resp.json().get("image_tag", "")
except Exception:
    ADAPTIVE_VERSION = ""

CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
VLLM_REF_TIMESTAMP = "2025-03-05-1720"
with open(f"{OUTPUT_DIR}/latest.txt") as f:
    ADAPTIVE_REF_TIMESTAMP = f.read().strip()
with open(f"{OUTPUT_DIR}/adaptive-latest-version.txt") as f:
    ADAPTIVE_REF_LATEST_VERSION = f.read().strip()

def check_run(cmd, **kwargs):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def get_locust_output_file(timestamp: str, provider: str, hit_cache: bool, is_lora: bool):
    cache_type = "perfectcache" if hit_cache else "nocache"
    subdir = f"{cache_type}_single_lora" if is_lora else cache_type
    return os.path.join(OUTPUT_DIR, subdir, f"{timestamp}-{provider}.csv")


def get_plotting_output_file(hit_cache: bool, exp:Literal["backbone_vs_adapter", "adaptive_vs_ref"]):
    cache_str = "perfectcache" if hit_cache else "nocache"
    if exp == "adaptive_vs_ref":
        return f"{OUTPUT_DIR}/reports/{CURRENT_TIMESTAMP}-adaptive-{ADAPTIVE_VERSION}-vs-{ADAPTIVE_REF_LATEST_VERSION}-{cache_str}.html"
    else:
        return f"{OUTPUT_DIR}/reports/{CURRENT_TIMESTAMP}-adaptive-lora-vs-backbone-{cache_str}.html"


def run_benchmark(provider, endpoint, api_key=None, hit_cache=True, is_lora=False):
    """
    Launch a benchmark for a given provider (adaptive/vllm), cache mode, and optional LoRA.
    """
    output_path = get_locust_output_file(CURRENT_TIMESTAMP, provider, hit_cache, is_lora)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    args = ["python", "launch_all.py", "-s", output_path, "-u", endpoint]

    assert provider == "adaptive", "Only adaptive provider is supported in this script"
    if provider == "adaptive":
        args += ["-p", "adaptive"]
        model_path = (
            "test"
            if not is_lora
            else "lora_benchmarking_test/adapt-llama-3-1-8b-finance-rag-2025-04-07-17-54"
        )
        args += ["-m", model_path]
        if api_key:
            args.append(f"-k={api_key}")

    if not hit_cache:
        args.append("-r")

    check_run(args)


def run_plot(
    model: str,
    output_tokens: int,
    input_files: list[str],
    output_file: str,
    extra_header=None,
    provider_suffixes=None,
):
    args = [
        "python",
        "plotting.py",
        "--model",
        model,
        "--output-tokens",
        str(output_tokens),
    ]

    if isinstance(input_files, list):
        args.append("--input-files")
        args += input_files

    args += ["--output-file", output_file]

    if extra_header:
        args += ["--extra-header", extra_header]

    if provider_suffixes:
        args += ["--provider-suffixes"] + provider_suffixes

    check_run(args)


adaptive_endpoint = os.environ.get("ADAPTIVE_ENDPOINT")
adaptive_api_key = os.environ.get("ADAPTIVE_API_KEY")
vllm_endpoint = os.environ.get("VLLM_ENDPOINT")

# Output for debugging
print("VLLM_ENDPOINT =", vllm_endpoint)
print("VLLM_REF =", VLLM_REF_TIMESTAMP)
print("ADAPTIVE_ENDPOINT =", adaptive_endpoint)
print("ADAPTIVE_API_KEY =", adaptive_api_key)
print("ADAPTIVE_VERSION =", ADAPTIVE_VERSION)
print("ADAPTIVE_REF =", ADAPTIVE_REF_TIMESTAMP)
print("ADAPTIVE_REF_LATEST_VERSION =", ADAPTIVE_REF_LATEST_VERSION)

for is_lora in [True, False]:
    for hit_cache in [True, False]:
        run_benchmark(
            provider="adaptive",
            endpoint=adaptive_endpoint,
            api_key=adaptive_api_key,
            hit_cache=hit_cache,
            is_lora=is_lora,
        )

# right now we don't have vllm loras so we do two loops
# loop 1, all comparisons with vllm and ref adaptive
for hit_cache in [True, False]:
    cache_str: Literal['perfectcache'] | Literal['nocache'] = "perfectcache" if hit_cache else "nocache"
    input_file_names = [
        get_locust_output_file(ts, p, hit_cache, False)
        for ts, p in [
            (CURRENT_TIMESTAMP, "adaptive"),
            (ADAPTIVE_REF_TIMESTAMP, "adaptive"),
            (VLLM_REF_TIMESTAMP, "vllm"),
        ]
    ]

    run_plot(
        model="Llama-3.1-8b",
        output_tokens=128,
        input_files=input_file_names,
        output_file=get_plotting_output_file(hit_cache, "adaptive_vs_ref"),
        provider_suffixes=[ADAPTIVE_VERSION, ADAPTIVE_REF_LATEST_VERSION, "0.7"],
        extra_header=f"Adaptive {ADAPTIVE_VERSION} vs. vllm 0.7.1 (randomized prompts)",
    )

for hit_cache in [True, False]:
    cache_str = "perfectcache" if hit_cache else "nocache"
    input_file_names = [get_locust_output_file(CURRENT_TIMESTAMP, "adaptive", hit_cache, is_lora) for is_lora in [False, True]]

    run_plot(
        model="Llama-3.1-8b",
        output_tokens=128,
        input_files=input_file_names,
        output_file=get_plotting_output_file(hit_cache, "backbone_vs_adapter"),
        provider_suffixes=["backbone", "lora"]
    )

# Save reference versions
with open(f"{OUTPUT_DIR}/latest.txt", "w") as f:
    f.write(CURRENT_TIMESTAMP)
with open(f"{OUTPUT_DIR}/adaptive-latest-version.txt", "w") as f:
    f.write(ADAPTIVE_VERSION)

url_messages = [ f"*Latest Inference Benchmark Reports Available: Adaptive [{ADAPTIVE_VERSION}] vs [{ADAPTIVE_REF_LATEST_VERSION}]*"]
for exp in ["backbone_vs_adapter", "adaptive_vs_ref"]:
    url_messages.append(exp)
    for is_cache_hit in [True, False]:
        nice_cache_msg = "Perfect Cache" if is_cache_hit else "No Cache"
        url = get_plotting_output_file(is_cache_hit, exp).replace(f"{OUTPUT_DIR}/reports", BENCHMARKS_PAGE) # type: ignore
        url_messages.append(f"🔗 <{url}|{nice_cache_msg}>")

print("Publish Slack message")
slack_payload = {
    "text": "\n".join(url_messages)
}
slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
if slack_webhook_url:
    requests.post(
        slack_webhook_url,
        headers={"Content-type": "application/json"},
        data=json.dumps(slack_payload),
    )
