import click
import subprocess
import logging
import threading
from pathlib import Path
import time
import pandas as pd
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def output_reader(pipe, prefix):
    for line in iter(pipe.readline, ""):
        line = line.strip()
        if line:
            logger.info(f"{prefix}: {line}")
    pipe.close()



def launch_model(
    name,
    url,
    api_key,
    model,
    tokenizer,
    provider,
    qps,
    duration,
    prompt_length,
    output_length,
    summary_file,
    randomize,
):
    """Launch a load test subprocess for a single model and return process and threads."""
    cmd = [
        "locust",
        "-H", url,
        "-m", model,
        "--tokenizer", tokenizer,
        "--provider", provider,
        "--qps", str(qps),
        "-u", "10",
        "-r", "10",
        "-p", str(prompt_length),
        "-o", str(output_length),
        "--chat",
        "--stream",
        "--summary-file", str(summary_file),
        "-t", str(duration),
        "-k", api_key,
    ]
    if randomize:
        cmd.append("--prompt-randomize")

    logger.info(f"Launching {name} at {qps} QPS → {summary_file}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    threads = [
        threading.Thread(target=output_reader, args=(process.stdout, name), daemon=True),
        threading.Thread(target=output_reader, args=(process.stderr, name), daemon=True),
    ]
    for t in threads:
        t.start()

    return process, threads


def combine_csv_files(files, output_file, delete_originals=True):
    """Combine multiple CSV files into a single CSV and optionally delete the originals."""
    dfs = []
    
    for file in files:
        if Path(file).exists():
            df = pd.read_csv(file)
            # Add file identifier as a column
            test_name = Path(file).stem
            df['Test'] = test_name
            
            # Add model name and QPS as separate columns for easier plotting
            if 'model1' in test_name:
                df['ModelName'] = 'Model 1'
                df['TargetQPS'] = df['Qps']  # Use actual QPS for model 1
            else:  # model2
                df['ModelName'] = 'Model 2'
                # Extract QPS from filename for model 2
                if 'qps' in test_name:
                    qps = int(test_name.split('_')[-1])
                    df['TargetQPS'] = qps
                else:
                    df['TargetQPS'] = df['Qps']
            
            dfs.append(df)
    
    if not dfs:
        logger.warning("No CSV files were found to combine")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataframe
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined data saved to {output_file}")
    
    # Optionally delete original files
    if delete_originals:
        for file in files:
            if Path(file).exists():
                os.remove(file)
                logger.info(f"Deleted original file: {file}")


@click.command()
@click.option("--url", required=True, help="API base URL (used for locust and keep-alive)")
@click.option("--api_key", required=True, help="API key")
@click.option("--summary-dir", default="results/", help="Directory where summary CSVs will be stored")
@click.option("--randomize", is_flag=True, help="Enable prompt randomization")
@click.option("--consolidated-output", default="benchmark_results.csv", help="Path for consolidated CSV output")
def run_benchmark(url, api_key, summary_dir, randomize, consolidated_output):
    """Run load tests for two models in parallel with a keep-alive thread to prevent port-forward disconnects."""
    # Config
    provider = "adaptive"
    duration = 120
    prompt_length = 1024
    output_length = 256
    tokenizer1 = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer2 = "Qwen/Qwen2.5-0.5B"
    model1 = "test"
    model2 = "test2"
    constant_qps = 4
    increasing_qps = [2, 4, 8]

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    threads = []
    summary_files = []

    summary_file1 = summary_dir / "model1_constant.csv"
    summary_files.append(summary_file1)
    p1, t1 = launch_model(
        "Model 1",
        url,
        api_key,
        model1,
        tokenizer1,
        provider,
        constant_qps,
        duration,
        prompt_length,
        output_length,
        summary_file1,
        randomize,
    )
    processes.append(p1)
    threads.extend(t1)

    for qps in increasing_qps:
        summary_file2 = summary_dir / f"model2_qps_{qps}.csv"
        summary_files.append(summary_file2)
        p2, t2 = launch_model(
            f"Model 2 qps({qps})",
            url,
            api_key,
            model2,
            tokenizer2,
            provider,
            qps,
            duration // len(increasing_qps),
            prompt_length,
            output_length,
            summary_file2,
            randomize,
        )
        processes.append(p2)
        threads.extend(t2)
        time.sleep(duration // len(increasing_qps)) 

    logger.info("🚀 All processes started. Waiting for completion...")

    for proc in processes:
        proc.wait()

    logger.info("✅ All benchmarks finished!")
    
    combine_csv_files(summary_files, consolidated_output, delete_originals=True)
    logger.info(f"Generated consolidated results in {consolidated_output}")


if __name__ == "__main__":
    run_benchmark()