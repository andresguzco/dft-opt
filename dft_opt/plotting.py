import seaborn as sns
import matplotlib.pyplot as plt    


def plot_energy(history, args):
    sns.set_theme(style="whitegrid")
    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total energy (Hartree)")
    plt.title("Energy Optimization of " + args.molecule)
    plt.savefig(f'plots/energy_{args.molecule}_{args.basis}_{args.optimizer}.png')
    plt.close()
