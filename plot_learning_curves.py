""" Script for aggregating learning curve csv files """

from pathlib import Path
import matplotlib.pyplot as plt

def parse_csv(csv_path:Path):
    """ """
    epoch,loss,mse,val_loss,val_mse = zip(*map(lambda l: map(float,l),
        [l.strip().split(",") for l in csv_path.open("r").readlines()][1:]))
    return {"epoch":epoch, "loss":loss, "mse":mse,
            "val_loss":val_loss, "val_mse":val_mse}

if __name__=="__main__":
    model_root_dir = Path("data/models")
    progress_csvs = [
            [p.name,next(f for f in p.iterdir() if f.name=="prog.csv")]
            for p in model_root_dir.iterdir()]

    ## Parse all the CSV files in the models directory tree
    models,curves = zip(*[(model,parse_csv(path))
        for model,path in progress_csvs])

    ## Collect common keys into a single dict mapping field keys to a list of
    ## models, which each contain an individual number of data points per epoch
    cdict = {k:[c[k] for c in curves] for k in curves[0].keys()}

    fig,ax = plt.subplots()
    cm = plt.get_cmap("gist_rainbow")
    #ax.set_prop_cycle(color=[cm(1.*i/len(models)) for i in range(len(models))])

    ## iterate over models plotting the above field
    for i in range(len(models)):
        color = cm(i//2*2.0/len(models))
        ax.plot(cdict["epoch"][i], cdict["mse"][i], label=models[i],
                color=color, linestyle="solid")
        ax.plot(cdict["epoch"][i], cdict["val_mse"][i], label=models[i],
                color=color, linestyle="dashed")
    ax.legend(ncol=3)
    ax.set_ylim([0,2])
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Training Epoch")
    plt.savefig("figures/lstmae_mse.png")
