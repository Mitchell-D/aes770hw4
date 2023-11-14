"""
Simple script that helps iterate over existing swath pkl data.

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

from krttdkit.operate import enhance as enh

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

    ## Some CERES values escape previous unmasking and default to a
    ## value about 3e38. Just get rid of any excessively large numbers.
    c_nonan = np.all(np.squeeze(ceres) < 1e30, axis=1)

    ## Mask any invalid modis pixels
    m_nonan = np.all(np.all(np.isfinite(modis), axis=1), axis=1)
    valid_count = np.count_nonzero(m_nonan)
    nancount = m_nonan.size-np.count_nonzero(m_nonan)
    if nancount > 0:
        mband_hasnan= set(np.where(np.logical_not(np.all(np.all(np.isfinite(
            modis[np.logical_not(m_nonan)]
            ), axis=0),axis=0)))[0])
        #mband_hasnan = modis[np.logical_not(m_nonan)].shape
        print(f"Found {nancount} NaN values in features " + \
                f"{mband_hasnan} {tmp_file.name} ")


    ## Collect the modis and ceres masks together and subset the data
    mutual_valid = np.logical_and(m_nonan, c_nonan)
    if np.count_nonzero(mutual_valid) == 0:
        return None
    modis = modis[mutual_valid]
    ceres = ceres[mutual_valid]
    print(f"t:{tmp_time} ceres:{ceres.shape} modis:{modis.shape} " + \
            f"d0:{int(np.average(modis[:,0,-2]))} f:{tmp_file.name}")
    pkl.dump(((clab,ceres),(mlab,modis)), tmp_file.open("wb"))

    mx = np.apply_over_axes(np.max, modis, (0,1))
    mn = np.apply_over_axes(np.min, modis, (0,1))
    av = np.apply_over_axes(np.mean, modis, (0,1))

    return mlab,mn,mx,av

if __name__=="__main__":
    agg_dir = Path("/rstor/mdodson/aes770hw4/validation")
    workers = 40
    #workers = 1

    agg_files = [f for f in agg_dir.iterdir()]
    terra = tuple((datetime.fromtimestamp(int(f.stem.split("_")[-1])), f)
            for f in agg_files if "terra" in f.name)
    aqua = tuple((datetime.fromtimestamp(int(f.stem.split("_")[-1])), f)
            for f in agg_files if "aqua" in f.name)

    #'''
    print("aqua 2015", len([f for t,f in aqua if 2015 == t.year]))
    print("aqua 2017", len([f for t,f in aqua if 2017 == t.year]))
    print("aqua 2021",len([f for t,f in aqua if 2021 == t.year]))
    print("terra 2015",len([f for t,f in terra if 2015 == t.year]))
    print("terra 2017",len([f for t,f in terra if 2017 == t.year]))
    print("terra 2021",len([f for t,f in terra if 2021 == t.year]))
    #'''
    #exit(0)

    avgs = []
    mins = []
    maxs = []
    with Pool(workers) as pool:
        swath_count = 0
        for result in pool.imap(get_mma, (*terra, *aqua)):
        #for result in pool.map(remove_nan, (*terra, *aqua)):
        #for result in pool.map(remove_nan, (*terra, *aqua)):
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