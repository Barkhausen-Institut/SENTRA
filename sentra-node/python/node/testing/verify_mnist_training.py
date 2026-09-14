
import subprocess
import sys
import time
import os

def verify_training():
    print("Verifying MNIST Secure Training...")
    
    # We will run 3 nodes in parallel using subprocess, piping output to files
    processes = []
    log_files = []
    
    try:
        for i in range(1, 4):
            log_file = open(f"node_{i}_verify.log", "w")
            log_files.append(log_file)
            
            cmd = [
                sys.executable,
                "run_mnist_batched_secure.py",
                "--node-id", str(i),
                "--n-nodes", "3",
                "--num-epochs", "1",
                "--batch-size", "4",
                "--mnist-samples", "16",
                "--learning-rate", "0.05",
                "--enable-network",
                "--loss-mode", "softmax",
                "--field-size", str(2**61 - 1),
                "--scale-factor", "65536",
                "--softmax-temperature", "2",
                "--exp-approx", "pade22",
                "--softmax-grad-mode", "opened_exact",
            ]
            
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append(p)
            
        print("Waiting for training to complete...")
        
        # Wait for all to finish (should be fast with 16 samples)
        start_time = time.time()
        timeout = 60 # seconds
        
        while True:
            if time.time() - start_time > timeout:
                print("Timeout waiting for training!")
                for p in processes:
                    p.terminate()
                return False
            
            all_done = all(p.poll() is not None for p in processes)
            if all_done:
                break
            time.sleep(1)
            
        print("Processes finished. Checking logs...")
        
        success_count = 0
        for i, log_file in enumerate(log_files):
            log_file.close()
            with open(f"node_{i+1}_verify.log", "r") as f:
                content = f.read()
                if "Epoch 1 Complete" in content:
                    success_count += 1
                else:
                    print(f"Node {i+1} failed to complete epoch 1. Log:\n{content}")
                    
        if success_count == 3:
            print("VERIFICATION SUCCESS: All nodes completed Epoch 1.")
            return True
        else:
            print(f"VERIFICATION FAILED: Only {success_count}/3 nodes succeeded.")
            return False
            
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for f in log_files:
            if not f.closed:
                f.close()
        # Cleanup logs
        # for i in range(1, 4):
        #     if os.path.exists(f"node_{i}_verify.log"):
        #         os.remove(f"node_{i}_verify.log")

if __name__ == "__main__":
    if verify_training():
        sys.exit(0)
    else:
        sys.exit(1)
