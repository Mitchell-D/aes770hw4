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
import zarr
import numpy as np
from datetime import datetime
import pickle as pkl
from multiprocessing import Pool

from krttdkit.operate import enhance as enh

from norm_vals import ceres_means, ceres_stdevs, modis_means, modis_stdevs

def get_mmm(swath):
    """
    Returns the MODIS labels, min, mean, and max of each feature within
    the provided swath. This is just a quick and dirty way of ensuring
    array validity.

    :return: (labels, mins, means, maxs)
    """
    tmp_time,tmp_file = swath
    ctup, mtup = pkl.load(tmp_file.open("rb"))
    _,ceres = ctup
    mlab,modis = mtup

    mx = np.apply_over_axes(np.max, modis, (0,1))
    mn = np.apply_over_axes(np.min, modis, (0,1))
    av = np.apply_over_axes(np.mean, modis, (0,1))

    return mlab,mn,av,mx

def remove_nan(swath):
    """
    If there are NaN values, remove them from each swath pkl by footprint
    and re-load the result back into the original pkl file

    :@param swath: 2-tuple like (datetime, file_path) to a pkl file containing
        a 2-tuple like ((ceres labels, ceres data), (modis_labels, modis_data))

    :@param return: 2-tuple like (ceres, modis) just containing the data lists.
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
        print(f"No valid data: {tmp_file}")
        return None
    modis = modis[mutual_valid]
    ceres = ceres[mutual_valid]
    print(f"t:{tmp_time} ceres:{ceres.shape} modis:{modis.shape} " + \
            f"d0:{int(np.average(modis[:,0,-2]))} f:{tmp_file.name}")

    ## Write back to the same pkl
    #print((ceres.shape,modis.shape), tmp_file.as_posix())
    pkl.dump(((clab,ceres),(mlab,modis)), tmp_file.open("wb"))

    ## Return the swath object and s
    return tmp_file

def mp_remove_nan(swath_list, workers=1):
    """
    """
    swath_count = 0
    with Pool(workers) as pool:
        for result in pool.map(remove_nan, swath_list):
            swath_count += 1
            #if result is None:
            #    continue
            #print(f"All valid: {result}")

def swath_size(swath):
    """
    Returns the Path and shapes of the ceres and modis arrays in the swath
    """
    ## Stored file is like ((ceres labels, ceres data),
    ##                      (modis_labels, modis_data))
    tmp_time,tmp_file = swath
    ctup, mtup = pkl.load(tmp_file.open("rb"))
    clab, ceres = ctup
    mlab, modis = mtup
    print(ceres.shape, modis.shape)
    try:
        assert np.all(np.isfinite(ceres))
        assert np.all(np.isfinite(modis))
    except:
        print(f"NaNs in {tmp_file}")
    return (tmp_file, ceres.shape, modis.shape)

def mp_swath_size(swath_list, workers=1):
    """ """
    swath_count = 0
    records = []
    with Pool(workers) as pool:
        for result in pool.imap(swath_size, swath_list):
            records += result
    return records

def mp_get_mmm(swath_list):
    avgs, mins, maxs = [], [], []
    with Pool(workers) as pool:
        swath_count = 0
        for result in pool.imap(get_mma, (*terra, *aqua)):
            mlab,mn,mx,av = result
            if result is None:
                continue
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

def swaths_to_zarr(
        swaths, ceres_path:Path, modis_path:Path,
        ceres_mean_array, modis_mean_array,
        ceres_stdev_array, modis_stdev_array,
        first_distance_cutoff=2000, overwrite=False
        ):
    """
    Iterates through the provided swath generator, and adds the CERES and MODIS
    data to separate zarr arrays.
    """
    if not overwrite:
        assert not ceres_path.exists()
        assert not modis_path.exists()

    C = None
    M = None
    ceres_store = zarr.ZipStore(ceres_path, mode="w")
    modis_store = zarr.ZipStore(modis_path, mode="w")
    for S in swaths:
        ctup,mtup = S
        clab,cdat = ctup
        mlab,mdat = mtup

        ## Mask out invalid distances (greater than 2km)
        m_valid_dist = mdat[:,0,mlab.index("dist")] < 2000
        cdat = cdat[m_valid_dist]
        mdat = mdat[m_valid_dist]

        ## Gauss-Scale all the data by the global means/stdevs
        cdat -= ceres_mean_array
        cdat /= ceres_stdev_array
        mdat -= modis_mean_array
        mdat /= modis_stdev_array

        ## Skip any swaths with no footprints in range
        if mdat.shape[0] == 0:
            continue

        ## Create the zarr arrays if they aren't yet initialized
        if M is None:
            C = zarr.creation.array(
                    data=cdat,
                    chunks=(1,*cdat.shape[1:]),
                    store=ceres_store,
                    )
            M = zarr.creation.array(
                    data=mdat,
                    chunks=(1,*mdat.shape[1:]),
                    store=modis_store,
                    )
            C.attrs["labels"] = clab
            M.attrs["labels"] = mlab
        ## Otherwise, append to the existing zarr array store
        else:
            C.append(cdat, axis=0)
            M.append(mdat, axis=0)
        ## Save the storage file and re-open the memory map
        #ceres_store.flush()
        #modis_store.flush()
    ceres_store.close()
    modis_store.close()
    return (C, M)

if __name__=="__main__":
    #agg_dir = Path("/rstor/mdodson/aes770hw4/validation")
    #ceres_zarr_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    #modis_zarr_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")
    #agg_dir = Path("/rstor/mdodson/aes770hw4/training")
    #ceres_zarr_path = Path("/rstor/mdodson/aes770hw4/ceres_training.zip")
    #modis_zarr_path = Path("/rstor/mdodson/aes770hw4/modis_training.zip")
    agg_dir = Path("/rstor/mdodson/aes770hw4/testing")
    ceres_zarr_path = Path("/rstor/mdodson/aes770hw4/ceres_testing.zip")
    modis_zarr_path = Path("/rstor/mdodson/aes770hw4/modis_testing.zip")
    workers = 1

    #'''
    """ Remove NaN values from all of the swaths in the aggregated pkls """
    swath_files = [f for f in agg_dir.iterdir()]
    terra = tuple((datetime.fromtimestamp(int(f.stem.split("_")[-1])), f)
            for f in swath_files if "terra" in f.name)
    aqua = tuple((datetime.fromtimestamp(int(f.stem.split("_")[-1])), f)
            for f in swath_files if "aqua" in f.name)
    swaths = (*terra, *aqua)
    mp_remove_nan(swath_list = (*terra, *aqua), workers=workers)
    #print(mp_swath_size(swaths, workers))
    #'''

    #'''
    """ Make gauss-normalized and distance-capped arrays of all swaths """
    cmean = np.array(list(zip(*ceres_means))[1])
    cstdev = np.array(list(zip(*ceres_stdevs))[1])
    mmean = np.array(list(zip(*modis_means))[1])
    mstdev = np.array(list(zip(*modis_stdevs))[1])

    #'''
    swath_paths = (p for p in agg_dir.iterdir() if "pkl" in p.name)
    swaths = (pkl.load(p.open("rb")) for p in swath_paths)
    swaths_to_zarr(
            swaths, ceres_zarr_path, modis_zarr_path,
            ceres_mean_array=cmean, modis_mean_array=mmean,
            ceres_stdev_array=cstdev, modis_stdev_array=mstdev,
            )
    #'''

    #ceres = zarr.Array(ceres_zarr_path, read_only=True)
    #modis = zarr.Array(modis_zarr_path, read_only=True)

    #X = modis[::12,:,-2]*mstdev[-2]+mmean[-2]
    #print(np.average(X), np.std(X))

    #'''
    """ use a subset of the data to estimate mean/stdev """
    '''
    ceres_labels = ceres.attrs["labels"]
    ceres = ceres[::15]
    print(ceres.shape, modis.shape)
    print(dict(ceres.attrs), dict(modis.attrs))
    print([(ceres_labels[i], np.average(ceres[:,:,i]))
        for i in range(ceres.shape[-1])])
    print([(ceres_labels[i], np.std(ceres[:,:,i]))
        for i in range(ceres.shape[-1])])

    modis_labels = modis.attrs["labels"]
    modis = modis[::15]
    print([(modis_labels[i], np.average(modis[:,:,i]))
        for i in range(modis.shape[-1])])
    print([(modis_labels[i], np.std(modis[:,:,i]))
        for i in range(modis.shape[-1])])
    #'''
