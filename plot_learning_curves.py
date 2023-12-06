""" Script for aggregating learning curve csv files """

from pathlib import Path
import matplotlib.pyplot as plt

def parse_csv(csv_path:Path):
    """ """
    #epoch,loss,mse,val_loss,val_mse = zip(*map(lambda l: map(float,l),
    #    [l.strip().split(",") for l in csv_path.open("r").readlines()][1:]))
    epoch,loss,val_loss= zip(*map(lambda l: map(float,l),
        [l.strip().split(",") for l in csv_path.open("r").readlines()][1:]))
    #return {"epoch":epoch, "loss":loss, "mse":mse,
    #        "val_loss":val_loss, "val_mse":val_mse}
    return {"epoch":epoch, "loss":loss, "val_loss":val_loss}

if __name__=="__main__":
    model_root_dir = Path("data/models")
    id_str = "lstmed"
    progress_csvs = [
            [p.name,next(f for f in p.iterdir() if f.name=="prog.csv")]
            for p in model_root_dir.iterdir() if id_str in p.name]

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
        color = cm(i/len(models))
        #ax.plot(cdict["epoch"][i], cdict["mse"][i], label=models[i],
        #ax.plot(cdict["epoch"][i], cdict["val_loss"][i], label=models[i],
        ax.plot(cdict["epoch"][i], cdict["loss"][i], label=models[i]+" loss",
                color=color, linestyle="solid",linewidth=1)
        ax.plot(cdict["epoch"][i], cdict["val_loss"][i], label=models[i]+" val_loss",
                color=color, linestyle="dashed", linewidth=1)
    ax.legend(ncol=2)
    ax.set_ylim([0,.35])
    #ax.set_title("Validation MSE Learning Curve")
    ax.set_title("Decoder Custom-loss Learning Curves")
    ax.set_xlim([0,450])
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Gaussian-normalized MSE")
    #plt.savefig("figures/lstmae_mse-val.png", dpi=800)
    plt.savefig("figures/lstmed_loss.png", dpi=800)
