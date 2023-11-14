"""
Simple script that evaluates existing swath pkl data.

Swath pkls are formatted like (platform)_(swath-epoch).pkl
where platform is "terra" or "aqua" and swath-epoch is the int timestamp.
and include a CERES footprint and co-located MODIS pixels like

( (ceres_labels, (N,1,C)),
  (modis_labels, (N,K,M) )

such that...

N: Number of valid footprints in the swath
K: Number of nearby modis pixels collected for each footprint
M: Number of MODIS bands
C: Number of CERES bands
"""
from pathlib import Path
import numpy as np
from datetime import datetime
import pickle as pkl
from multiprocessing import Pool
def get_mma(swath):
    tmp_time,tmp_file = swath
    ctup, mtup = pkl.load(tmp_file.open("rb"))
    _,ceres = ctup
    mlab,modis = mtup

    mx = np.apply_over_axes(np.max, modis, (0,1))
    mn = np.apply_over_axes(np.min, modis, (0,1))
    av = np.apply_over_axes(np.mean, modis, (0,1))

    return mlab,mn,mx,av

def remove_nan(swath):
    """
    If there are NaN values, remove them from each swath pkl by footprint
    and re-load the result back into the pkl file
    """
    tmp_time,tmp_file = swath
    ctup, mtup = pkl.load(tmp_file.open("rb"))
    clab,ceres = ctup
    mlab,modis = mtup

    nonan = np.all(np.all(np.isfinite(modis), axis=1), axis=1)
    if np.any(np.logical_not(nonan)):
        nancount = nonan.size-np.count_nonzero(nonan)
        print(f"Found NaN: {nancount} in {tmp_file.name}")
        print([
            np.average(modis[np.logical_not(nonan)][...,i])
            for i in range(len(mlab))
            ])
    if np.count_nonzero(nonan) == 0:
        print(f"No valid data!: {tmp_file.name}")
        return None
    modis = modis[nonan]
    ceres = ceres[nonan]
    print(f"Writing pkl to {tmp_file.name}")
    pkl.dump(((clab,ceres),(mlab,modis)), tmp_file.open("wb"))

    mx = np.apply_over_axes(np.max, modis, (0,1))
    mn = np.apply_over_axes(np.min, modis, (0,1))
    av = np.apply_over_axes(np.mean, modis, (0,1))

    return mlab,mn,mx,av

if __name__=="__main__":
    agg_dir = Path("/rstor/mdodson/agg_ceres_modis")
    workers = 30

    agg_files = [f for f in agg_dir.iterdir()]
    terra = tuple((datetime.fromtimestamp(int(f.stem.split("_")[-1])), f)
            for f in agg_files if "terra" in f.name)
    aqua = tuple((datetime.fromtimestamp(int(f.stem.split("_")[-1])), f)
            for f in agg_files if "aqua" in f.name)

    #'''
    print(len([f for t,f in aqua if 2015 == t.year]))
    print(len([f for t,f in aqua if 2017 == t.year]))
    print(len([f for t,f in aqua if 2021 == t.year]))
    print(len([f for t,f in terra if 2015 == t.year]))
    print(len([f for t,f in terra if 2017 == t.year]))
    print(len([f for t,f in terra if 2021 == t.year]))
    #'''
    exit(0)

    avgs = []
    mins = []
    maxs = []
    with Pool(workers) as pool:
        swath_count = 0
        #for result in pool.imap(get_mma, (*terra, *aqua)):
        for result in pool.imap(remove_nan, (*terra, *aqua)):
            if result is None:
                continue
            mlab,mn,mx,av = result
            try:
                avgs.append(av)
                mins.append(mn)
                maxs.append(mx)
            except Exception as e:
                print(e)
            finally:
                swath_count += 1

    avgs = np.concatenate(avgs, axis=0)
    mins = np.concatenate(mins, axis=0)
    maxs = np.concatenate(maxs, axis=0)

    stats = [(
        mlab[i],
        f"min mam: {np.min(mins[...,i]):2f} "
        f"{np.average(mins[...,i]):2f} {np.max(mins[...,i]):2f}",
        f"avg mam: {np.min(avgs[...,i]):2f} "
        f"{np.average(avgs[...,i]):2f} {np.max(avgs[...,i]):2f}",
        f"max mam: {np.min(maxs[...,i]):2f} "
        f"{np.average(maxs[...,i]):2f} {np.max(maxs[...,i]):2f}",
        ) for i in range(len(mlab))]
    for s in stats:
        print(s)
