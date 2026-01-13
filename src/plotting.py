import matplotlib.pyplot as plt

def plot_loss_histories(histories, title, save_path=None, lr_info=None):
    plt.figure(figsize=(10, 6))

    for name, losses in histories.items():
        plt.plot(losses, label=name)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Iterations (Updates)")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    else:
        plt.show()
