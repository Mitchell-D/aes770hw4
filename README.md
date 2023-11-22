# aes770hw4

## Charge

The final project must use satellite data and combination of
radiative transfer/Machine learning models to address an
earth-atmosphere problem.

## Initial concept / overview

<p align="center">
  <img width="800" src="https://github.com/Mitchell-D/aes770hw4/blob/main/figures/initial_concept.jpg" />
</p>

## Software Goals

 1. Download MODIS L1b imager data and CERES SSF longwave and
    shortwave fluxes within a region covering the South East US.
 2. For each of N CERES footprints in a satellite overpass (swath),
    collect the nearest K MODIS imager pixels. Encode their position
    as the great circle distance and azimuth to the CERES centroid.
 3. Store each swath as a (N,1,C) array of C CERES features, and a
    (N,K,M) array of M MODIS features, each along with a list of
    strings labeling each feature axis.
 3. Quality-check the data by making sure there are no NaN values or
    data outside of the acceptable range (CERES footprints sometimes
    escape masking and end up with a ~3e38 value). Remove any swaths
    that have no valid data from the dataset.
 5. Gauss-normalize all in-range data and collect it into zarr arrays
    for training and validation data
 6. Develop an LSTM based autoencoder architecture that reproduces a
    sequence of up to K of the nearest imager pixels, optimizing for
    spatial interpolation of masked values within the 400px domain.
 7. Develop an autoencoder architecture optimizing for spectral
    interpolation of masked values.
 8. Develop a decoder head that converts the latent-space output of
    both above models into an estimate of the longwave and shortwave
    *flux contribution* of each modis pixel, using a custom loss
    function that compares the spatial average to the bulk CERES
    footprint values for flux.

## Data Input

The autoencoder is initially only trained on MODIS pixels, even
though they are grouped in local sets by relating them to the CERES
gris. The following are the extracted MODIS bands, in order, followed
by the default order of MODIS geometry fields (appended later)

```python
##                      # MODIS Band Selection (in order)
modis_bands = [
        8,              # .41                       Near UV
        1,4,3,          # .64, .55, .46             (R,G,B)
        2,              # .86                       NIR
        18,             # .935                      NIR water vapor
        5,26,6,7,       # 1.24, 1.38, 1.64, 2.105   SWIR (+cirrus)
        20,             # 3.7                       Magic IR
        27,28,          # 6.5, 7.1                  (high,low) pwv
        30,             # 9.7                       ozone
        31,             # 10.9                      clean window
        33,             # 13.3                      co2
        ]               # lat, lon, height,         geodesy
##                      # sza, saa, vza, vaa        geometry
```

Although most aren't used, the following CERES fields are extracted
along with the geolocation and fluxes as context for future analysis
wrt surface types and cloud/aerosol forcing.

```python
labels = [ ## custom label mappings for CERES bands
        'jday', 'lat', 'lon', 'vza', 'raa', 'sza',   # Context
        'id_s1', 'id_s2', 'id_s3', 'id_s4',          # Sfc types
        'id_s5', 'id_s6', 'id_s7', 'id_s8',
        'pct_s1', 'pct_s2', 'pct_s3', 'pct_s4',      # Sfc dist
        'pct_s5', 'pct_s6', 'pct_s7', 'pct_s8',
        'pct_clr', 'pct_l1', 'pct_l2', 'pct_ol',
        'swflux', 'wnflux', 'lwflux',                # Fluxes
        'nocld', 'nocld_wk',                         # Cloud mask
        'l1_cod', 'l2_cod', 'l1_sdcod', 'l2_sdcod',  # Cloud layers
        'aer_land_pct', 'aer_land_cfrac',            # Land aero
        'aer_land_type', 'aod_land',
        'aer_db_pct', 'aod_db',                      # Deep blue aero
        'aer_ocean_pct', 'aer_ocean_cfrac',          # Ocean aero
        'aod_ocean', 'aod_ocean_small'
        ]
```

## Software

### `FG1D.py`

[FeatureGrid][2]-like class for basic operations on 1d arrays of features.
This will eventually be more generalized with the current (2D)
FeatureGrid and HyperGrid
classes

[2]:https://github.com/Mitchell-D/krttdkit/blob/main/krttdkit/products/FeatureGrid.py

### `get_ceres_swath.py`

Opens a [CERES SSF][3] file, extracts a series of its data features
(listed below), applies constraints to the footprints (ie sza, vza),
converts times to epoch, and divides footprints into individual
satellite overpasses by clustering their encoded times.

Stores the result as a 2-tuple like `(ceres_labels, ceres_data)`
where `ceres_labels` is a list of unique string labels corresponding
to uniform-size 1D arrays in `ceres_data`.

[3]:https://ceres.larc.nasa.gov/data/

### `get_modis_swath.py`

Extracts overpass time ranges of valid swaths from the CERES dataset,
and queries the [LAADS DAAC][1] to download the corresponding MODIS
L1b data granules (often multiple files per overpass). A series of
bands (listed below) are extracted, and are converted into
reflectance or brightness temperatures.

MODIS pixel values that are within the configured geodetic domain are
extracted as a list of 1-D arrays corresponding to each MODIS
feature. Note that these arrays have **not** yet been associated
with CERES data; they simply contain all MODIS pixels in range of the
pre-defined geographic area.

Ultimately, each swath's data is stored as pkl files containing a
list of labels and a list of arrays for both CERES and MODIS like:

```
((ceres_labels, ceres_data), (modis_labels, modis_data))
```

so that `ceres_data` is a (N,C) shaped array for N footprints and
C CERES features, and `modis_data` is a (P,M-2) shaped array for
P MODIS pixels and M MODIS features.

[1]:https://ladsweb.modaps.eosdis.nasa.gov/about/

### `aggregate_ceres_modis.py`

Loads CERES+MODIS data from the pkl output of `get_modis_swath.py`.
For each CERES footprint, calculates the great circle distance to
all MODIS pixels in the swath, and extracts the K pixels closest to
the footprint centroid (K is a configured value, initially 400),
then calculates the North-relative geodetic azimuth from the centroid
to each of the K pixels.

Ultimately, the MODIS pixels are collected as a (N,K,M) shaped array,
where M now includes 2 additional features corresponding to the
great circle distance and azimuth to each footprint centroid.

### `iter_agg.py`

Script for providing methods of iterating over aggregated CERES+MODIS
swath pkl data in order to calculate statistics, modify swath pkl
files by removing invalid footprints, or to construct zarr arrays
by merging all of the swaths.

The NaN checking in this file should eventually be migrated into the
`aggregate_ceres_modis.py` process pipeline.

### `lstm_ae.py`

Contains methods to dynamically create stacked LSTM autoencoders,
generator functions for providing shuffled footprints to the fit
method, and procedural code for actually doing the training.

### `crossval.py`

Contains probably-valid but currently-broken methods for doing
cross-validation with keras-tuner. There are some issues with keras-
tuner backend for which I haven't found an environment workaround.

