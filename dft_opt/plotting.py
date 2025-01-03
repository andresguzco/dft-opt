import seaborn as sns
import matplotlib.pyplot as plt    

def plot_energy(history, mol_name, method, optimizer):
    sns.set_theme(style="whitegrid")
    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total energy (Hartree)")
    plt.title("Energy Optimization of" + mol_name)
    plt.savefig(f'plots/energy_{optimizer}_{method}.png')
    plt.close()
    return None

def plot_batched_results(history, rs, E_total):
    sns.set_theme(style="whitegrid")

    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Batched Loss (Hartree)")
    plt.title("Optimization Loss History")
    plt.savefig('plots/loss.png')
    plt.close()

    plt.figure()
    ax = sns.lineplot(x=rs, y=E_total)
    ax.set_xlabel("$H_2$ bond length (a.u.)")
    ax.set_ylabel("Total Energy (Hartree)")
    plt.title("Bond Length vs Total Energy")
    plt.savefig('plots/distance.png')
    plt.close()