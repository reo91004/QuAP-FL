import subprocess
import sys
import os
import time
from datetime import datetime

def run_experiment(mode, output_dir, dataset='mnist', rounds=50, seed=42):
    print(f"Running experiment: Mode={mode}, Dataset={dataset}, Rounds={rounds}")
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), '..', 'main.py'),
        '--dataset', dataset,
        '--mode', mode,
        '--num_rounds', str(rounds),
        '--seed', str(seed),
        '--output_dir', output_dir
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running {mode}:")
        print(result.stderr)
    else:
        print(f"Finished {mode} in {end_time - start_time:.2f}s")
        # Extract final accuracy from stdout
        for line in result.stdout.split('\n'):
            if "최종 정확도:" in line:
                print(f"  {line.strip()}")

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run QuAP-FL Experiments')
    parser.add_argument('--rounds', type=int, default=30, help='Number of rounds per experiment')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'smoke', 'fedavg', 'fixed_dp', 'quap_fl'], help='Experiment mode')
    args = parser.parse_args()

    # Create timestamped directory for this session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # scripts/에서 실행되므로 상위 폴더의 results/에 저장
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', timestamp)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Experiment Session: {timestamp} ===")
    print(f"Output Directory: {output_dir}")

    modes = ['fedavg', 'fixed_dp', 'quap_fl']
    
    if args.mode == 'smoke':
        print("\n=== Starting Smoke Tests (2 rounds) ===")
        for mode in modes:
            run_experiment(mode, output_dir, rounds=2)
        return

    if args.mode != 'all':
        modes = [args.mode]

    print(f"\n=== Starting Experiments ({args.rounds} rounds) ===")
    for mode in modes:
        run_experiment(mode, output_dir, rounds=args.rounds)
        
    print(f"\nAll experiments completed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
