# Nepal dataset extraction pipeline

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import h5py
import ee 
import geemap
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

GEE_PROJECT_ID = "nepal-landslide-extraction"

ARNIKO_SHP      = ""
LANGTANG_SHP    = ""
ASM_SHP         = ""

ASM_DATE_FIELD = "Post_Sat"

TILE_PX         = 128
TILE_RES_M      = 10
TILE_SIZE_M     = TILE_PX * TILE_RES_M

OUTPUT_IMG_DIR  = "output/img"
OUTPUT_MASK_DIR = "output/masks"

S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  'B11', 'B12']
# 0:B1 1:B2(blue) 2:B3(green) 3:B4(red) 4:B5 5:B6 6:B7 7:B8(nir) 8:B8A 9:B9
# 10:B11(swir1) 11:B12(swir2) 12:slope 13:dem

# Step 1 - Load, standardize, and merge the three inventories

def load_inventories():
    arniko      = gpd.read_file(ARNIKO_SHP)
    langtang    = gpd.read_file(LANGTANG_SHP)
    asm         = gpd.read_file(ASM_SHP)
    
    arniko["source_inv"]    = "arniko"
    langtang["source_inv"]  = "langtang"
    asm["source_inv"]       = "asm"

    # arniko and langtang are already from 2017-2018, fully in the sentinel-2 era.
    #ASM spans from 1988-2018 - filter to sentinel-2-era entries only
    
    asm[ASM_DATE_FIELD] = pd.to_datetime(asm[ASM_DATE_FIELD], errors="coerce")
    asm_recent = asm[asm[ASM_DATE_FIELD] >= "2016-01-01"].copy()
    
    print(f"Araniko: {len(arniko)} | Langtang: {len(langtang)} "
          f"ASM : {len(asm_recent)} of {len(asm)} total")
    
    # Reconciling to common CRS before merging
    # here the target_crs defaults to the EPSG:4326 (WGS 84 latitude/longitude).
    # in this coordinates are expressed in degrees
    target_crs = arniko.crs or "EPSG:4326"
    langtang = langtang.to_crs(target_crs)
    asm_recent = asm_recent.to_crs(target_crs)
    
    combined = pd.concat (
          [arniko[["geometry", "source_inv"]],
           langtang[["geometry", "source_inv"]],
           asm[["geometry", "source_inv"]]],
           ignore_index=True
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=target_crs)
    return combined

# step-2 Build a square tile geometry around each landslide
# Here UTM Zone 45N (EPSG:32645) is the specific projected metric grid designed for Nepal and Northeast india# because it covers nepal on a flat grid measured in meters, calculating distances, bounding boxes, and padding around landlsides in Zone 45N is mathmatically exact
def make_tile_geometry(geom, size_m=TILE_SIZE_M, metric_crs="EPSG:32645"):
      # converts the landslide shape from degrees EPSG:4326 to meters EPSG:32645
      g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
      # Extract center point coordinates (cx,cy) in METERS
      cx, cy = g.centroid.x, g.centroid.y
      #compute half width in meters
      half = size_m / 2
      # construct a square box geometry in METERS
      tile_m = box(cx - half, cy - half, cx + half, cy + half)
      # Reproject the square box back to degrees for google earth engine
      tile_ll = gpd.GeoSeries([tile_m], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
      return tile_ll

# STEP - 3 - Pull Sentinel-2 + DEM + slope for one tile via Earth Engine

def get_stacked_image(tile_geom, center_date, window_days=45, max_cloud=30):
      # translates the python bounding box into a Google Earth Engine (EE) Rectangle. Tells the google servers exactly where on Earth to look
      ee_geom = ee.Geometry.Rectangle(list(tile_geom.bounds))
      
      # constructing a temporal search window 
      # window_days define a time duration (e.g., 45 days)
      # start and end calculates the date 45 days before the event and after the event
      #.strftime('%Y-%m-%d'): Formats the calculated date object into a clean string format required by earth Engine
      start = (pd.Timestamp(center_date - pd.Timedelta(days=window_days))).strftime('%Y-%m-%d')
      end = (pd.Timestamp(center_date + pd.Timedelta(days=window_days))).strftime('%Y-%m-%d')
      
      
      # it accesses earths engine full catalog of sentinel-2
      # .filterBounds(ee_geom): Spatial filter-restricts the search to only scenes that physically overlap with your bounding box (ee_geom)
      # fiterDate(start, end) : temporal filter-Limits the search to image take between the start date and the end date
      #last filter checks the metadata of each image and discards any image where the overall cloud percentage is greater or equal than max_cloud
      #.sort- orders the remaining iage in ascending order of 
      s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterBounds(ee_geom)
                 .filterDate(start, end)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
                 .sort("CLOUDY_PIXEL_PERCENTAGE")) 
      
      
      #.first(): extracts the very first image from sorted collection-which is the image with the lowest cloud percentage
      s2_image = s2_coll.first()
      if s2_image is None:
            return None
      
      # only select the s2 bands and ignore the others
      s2_selected =s2_image.select(S2_BANDS)
      #loads the JAXA ALOS World 3D(30m resolution) dataset
      #picks the digital surface model band
      # clip(ee_geom): crops he huge global elevation dataset down to the exact 1280x1280m tile
      dem = ee.Image("JAXA/ALOS/AW3D30/V3_2").select("DSM").clip(ee_geom)
      # uses earth eingine's builtin ee.Terrain.slope() algo
      slope = ee.Terrain.slope(dem)
      
      # Glues i band slope and dem to the 12 sentinel-2 optical bands using addbands()
      stacked = s2_selected.addBands(slope.rename("slope")).addbands(dem.rename("dem"))
          
      return stacked, ee_geom  
      
def download_tile_array(stacked_image, ee_geom, out_path, scale=TILE_RES_M):
      geemap.ee_export_image(
            stacked_image,
            file_name=out_path,
            scale=scale,
            region=ee_geom,
            file_per_band=False
      )

# Step - 4 - Rasterize the mask for a tile

# landslide_pols: A list of shapely polygons geometrics representing all landslides overlapping the region
# tif_path: The file path to the temporary satelite GeoTIFF image
def rasterize_mask_for_tile(landslide_polys, tif_path):
      # opens the downloaded satelite image, extracts the affine transform matrix which defines the exact geographical placement of the earth
      #shape gets the exact pixel dimesnion of the tile
      with rasterio.open(tif_path) as src:
            transform = src.transform
            shape = (src.height, src.width)
      
      # the rasterize function
      mask = rasterize(
            # pairs each landslide polygon shape with a "burn value" of 1. Every pisel that falls inside the shape will be assigned 1
            [(geom, 1) for geom in landslide_polys],
            out_shape = shape, # forces the mask array to have the exact same shape
            transform=transform, # uses the satellite images spatial matrix so that every pisel in the mask lines up with the corresponding pisel in satellite image
            fill = 0, # sets the default of all the values outside the landslide0
            dtype="uint8" # temporarily uses 8-bit integer to save memory during the process
      )
      return mask.astype(np.int64) # converts the final to 64 bit

