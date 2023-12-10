from pathlib import Path
#from swath_shapes import swath_validation, swath_training
import pickle as pkl

from FG1D import FG1D
import numpy as np
import os

#from norm_vals import ceres_means, ceres_stdevs, modis_means, modis_stdevs

if __name__=="__main__":
    #swath_out_dir = Path("/rstor/mdodson/aes770hw4/output")
    swath_out_dir = Path("/rstor/mdodson/aes770hw4/output_1")
    #tmp_swath = swath_out_dir.joinpath("pred_1566768237.pkl")

    for swath in swath_out_dir.iterdir():
        ceres, pcoarse, pfine = map(
                lambda t: FG1D(*t),
                pkl.load(swath.open("rb"))
                )
        _,epoch = swath.stem.split("_")
        plot_spec = {"marker_size":7}
        plot_spec["title"] = f"Epoch {epoch} CERES Shortwave Flux"
        ceres.geo_scatter(
                clabel="sw",
                fig_path=Path(f"figures/output_1/{epoch}_ceres_sw.png"),
                plot_spec=plot_spec,
                show=False,
                )
        plot_spec["title"] = f"Epoch {epoch} CERES Longwave Flux"
        ceres.geo_scatter(
                clabel="lw",
                fig_path=Path(f"figures/output_1/{epoch}_ceres_lw.png"),
                plot_spec=plot_spec,
                show=False,
                )
        '''
        pfine.geo_scatter(
                clabel="sw",
                fig_path=Path(f"figures/output_1/{epoch}_fine_sw.png"),
                )
        pfine.geo_scatter(
                clabel="lw",
                fig_path=Path(f"figures/output_1/{epoch}_fine_lw.png"),
                )
        '''
        plot_spec["title"] = f"Epoch {epoch} Model Shortwave Flux"
        pcoarse.geo_scatter(
                clabel="sw",
                fig_path=Path(f"figures/output_1/{epoch}_coarse_sw.png"),
                plot_spec=plot_spec,
                show=False,
                )
        plot_spec["title"] = f"Epoch {epoch} Model Longwave Flux"
        pcoarse.geo_scatter(
                clabel="lw",
                fig_path=Path(f"figures/output_1/{epoch}_coarse_lw.png"),
                plot_spec=plot_spec,
                show=False,
                )
