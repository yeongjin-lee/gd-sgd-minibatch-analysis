import os
from experiments.exp1_loss_reduction import run_exp1
from experiments.exp2_noise_vs_batch import run_exp2

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    print("Running Experiment 1...")
    run_exp1()

    print("\nRunning Experiment 2...")
    run_exp2()
