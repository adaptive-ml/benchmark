#!/usr/bin/env python3
import argparse
import os
import subprocess
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run locust load tests.')
    parser.add_argument('-p', '--provider', default='vllm', help='Provider name')
    parser.add_argument('-s', '--summary-file', dest='summary_file', default='vllm.csv',
                        help='Summary file path')
    parser.add_argument('-u', '--url', required=True, help='Target URL for locust tests')
    parser.add_argument('-m', '--model', default='meta-llama/Llama-3.1-8B-Instruct', help='Model name')
    parser.add_argument('-k', '--api-key', dest='api_key', default='not-relevant-for-vllm', help='API key')
    parser.add_argument('-r', '--randomize', action='store_true', help='Randomize prompts')
    parser.add_argument('-d', '--duration', type=int, default=60, help='Test duration in seconds')
    return parser.parse_args()


def main():
    args = parse_args()

    # Read environment variables or use defaults
    # lengths_str = os.environ.get('LENGTHS', '128,256,512,1024,2048,4096')
    # qps_str = os.environ.get('QPS', '0.125,0.5,1,2,4,6,8,10,12,14,16,18,20')

    lengths_str = os.environ.get('LENGTHS', '256,1024')
    qps_str = os.environ.get('QPS', '0.125,0.5,1,2')

    lengths = lengths_str.split(',')
    qps = qps_str.split(',')

    print(args.duration)
    print(lengths_str)
    print(qps_str)

    for length in lengths:
        for q in qps:
            print(f"Running load test with {length} input token size and {q} qps\n")
            cmd = [
                'locust',
                '-H', args.url,
                '-m', args.model,
                '--tokenizer', 'meta-llama/Llama-3.1-8B-Instruct',
                '--provider', args.provider,
                '--qps', q,
                '-u', '500',
                '-r', '500',
                '-p', length,
                '-o', '128',
                '--chat',
                '--stream',
                '--summary-file', args.summary_file,
                '-t', str(args.duration),
                '-k', args.api_key
            ]
            if args.randomize:
                cmd.append('--prompt-randomize')
                
            if "CF_TOKEN" in os.environ:
                print("Adding CF_TOKEN to command")
                cmd.extend(['--cf-access-token', os.environ["CF_TOKEN"]])

            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            time.sleep(5)

if __name__ == '__main__':
    main()
