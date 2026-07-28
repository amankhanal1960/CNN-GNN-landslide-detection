# Nepla dataset extraction pipeline

import os
import pandas as pd
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

def make_tile_geometry(geom, size_m=TILE_SIZE_M, metric_crs="EPSG:32645"):
      """Centered square tile in lat/lon, sized to cover the polygon + margin"""  
      g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
      cx, cy = g.centroid.x, g.centroid.y
      half = max(size_m, (max(g.bounds[2]-g.bounds[0], g.bounds[3]-g.bounds[1]) + 200)) / 2
      tile_m = box(cx - half, cy - half, cx + half, cy + half)
      tile_ll = gpd.GeoSeries([tile_m], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
      return tile_ll

