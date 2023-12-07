from pathlib import Path
#from swath_shapes import swath_validation, swath_training
import pickle as pkl

from FG1D import FG1D
import numpy as np
import os

#from norm_vals import ceres_means, ceres_stdevs, modis_means, modis_stdevs

'''
ceres_means = dict(ceres_means)
ceres_stdevs = dict(ceres_stdevs)
modis_means = dict(modis_means)
modis_stdevs = dict(modis_stdevs)
'''
if __name__=="__main__":
    swath_out_dir = Path("/rstor/mdodson/aes770hw4/output")
    tmp_swath = swath_out_dir.joinpath("pred_1566768237.pkl")

    ceres, pcoarse, pfine = map(
            lambda t: FG1D(*t),
            pkl.load(tmp_swath.open("rb"))
            )

    #'''
    ceres.geo_scatter("sw", fig_path=Path("figures/ceres_sw.png"))
    ceres.geo_scatter("lw", fig_path=Path("figures/ceres_lw.png"))
    pfine.geo_scatter("sw", fig_path=Path("figures/fine_sw.png"))
    pfine.geo_scatter("lw", fig_path=Path("figures/fine_lw.png"))
    pcoarse.geo_scatter("sw", fig_path=Path("figures/coarse_sw.png"))
    pcoarse.geo_scatter("lw", fig_path=Path("figures/coarse_lw.png"))
    #'''
