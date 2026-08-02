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
from shapely import is_empty
from shapely.geometry import box
from pathlib import Path

GEE_PROJECT_ID = "nepal-landslide-extraction"

root = Path("E:\\Major project")
ASM_SHP = (root/"Datasets"/"asian"/"DataFile1 - 12838ASMInventory.shp")
LANGTANG_SHP = (root/"Datasets"/"lantang-valley"/"DataFile3_-_LangtangInventory.shp")
ARNIKO_SHP = (root/"Datasets"/"Araniko"/"DataFile4_-_ArnikoInventory.shp")

ASM_DATE_FIELD = "parsed_post_date"

def parse_landslide_scene_date(scene_id):
      try:
            parts = str(scene_id).split("_")
            year, month, day = int(parts[-3]), int(parts[-2]), int(parts[-1])
            return pd.Timestamp(year=year, month=month, day=day)
      except (ValueError, IndexError):
            return pd.NaT

TILE_PX         = 128
TILE_RES_M      = 10
TILE_SIZE_M     = TILE_PX * TILE_RES_M

OUTPUT_IMG_DIR  = root/"output/img"
OUTPUT_MASK_DIR = root/"output/masks"

S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',  'B11', 'B12']
# 0:B1 1:B2(blue) 2:B3(green) 3:B4(red) 4:B5 5:B6 6:B7 7:B8(nir) 8:B9 9:B10
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
    
    #.aaply loops through every row in the Post_Sat column and applies the parse_landslide_scene_date function to extract the date from the scene ID which is Post_Sat. The result is stored in a new column called parsed_post_date
    asm[ASM_DATE_FIELD] = asm["Post_Sat"].apply(parse_landslide_scene_date)
    asm_years = pd.to_numeric(asm["Year"], errors="coerce")
    asm_recent = asm[asm_years >= 2016].copy()
    
    print(f"Araniko: {len(arniko)} | Langtang: {len(langtang)} "
          f"ASM : {len(asm_recent)} of {len(asm)} total")
    
    # Reconciling to common CRS before merging
    # here the target_crs defaults to the EPSG:4326 (WGS 84 latitude/longitude) if target_crs is not specified.
    # in this coordinates are expressed in degrees
    target_crs = arniko.crs or "EPSG:4326"
    langtang = langtang.to_crs(target_crs)
    asm_recent = asm_recent.to_crs(target_crs)
    
    # since for langtang and arniko are from a single time period, we can fallback to a default date of 2017-11-01 for the center of the temporal window
    
    #Adds empty "parsed_post_date" columns to arniko and langtang so all three tables can be cleanly merged together into combined.
    keep_cols = ["geometry", "source_inv", ASM_DATE_FIELD]
    arniko[ASM_DATE_FIELD] = pd.NaT
    langtang[ASM_DATE_FIELD] = pd.NaT
    
    combined = pd.concat (
          [arniko[keep_cols],
           langtang[keep_cols],
           asm_recent[keep_cols]],
           ignore_index=True
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=target_crs)
    
    # Drop bad geometry (null/empty/invalid) before they crash the system
    before = len(combined)
    combined = combined[combined.geometry.notna()]
    combined = combined[~combined.geometry.is_empty]
    combined = combined[combined.geometry.is_valid]
    dropped = before - len(combined)
    if dropped:
              print(f"Dropped {dropped} invalid geometries")
    return combined

# step-2 Build a square tile geometry around each landslide
# Here UTM Zone 45N (EPSG:32645) is the specific projected metric grid designed for Nepal and Northeast india# because it covers nepal on a flat grid measured in meters, calculating distances, bounding boxes, and padding around landlsides in Zone 45N is mathmatically exact
def make_tile_geometry(geom, src_crs, size_m=TILE_SIZE_M, metric_crs="EPSG:32645"):
      # converts the landslide shape from degrees EPSG:4326 to meters EPSG:32645
      g = gpd.GeoSeries([geom], crs=src_crs).to_crs(metric_crs).iloc[0]
      if g.is_empty or not g.is_valid:
            raise ValueError("Invalid geometry provided for tile creation.")
      # Extract center point coordinates (cx,cy) in METERS
      cx, cy = g.centroid.x, g.centroid.y
      #compute half width in meters
      half = size_m / 2
      # construct a square box geometry in METERS
      # this places the centroid of the landslide exactly in the center of the frame
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
      s2_coll = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                 .filterBounds(ee_geom)
                 .filterDate(start, end)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
                 .sort("CLOUDY_PIXEL_PERCENTAGE")) 
      
      
      #.first(): extracts the very first image from sorted collection-which is the image with the lowest cloud percentage
      s2_image = s2_coll.first()
      if s2_image is None:
            return None
      
      # only select the s2 bands and ignore the others
      s2_selected = s2_image.select(S2_BANDS)
      #loads the JAXA ALOS World 3D(30m resolution) dataset
      #picks the digital surface model band
      # clip(ee_geom): crops he huge global elevation dataset down to the exact 1280x1280m tile
      dem = (ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1")
            .filterBounds(ee_geom)
            .select("DSM")
            .mosaic()
      )
      slope = ee.Terrain.slope(dem)
      stacked = (s2_selected
                 .addBands(slope.rename("slope"))
                 .addBands(dem.rename("dem"))
                 .resample("bilinear")
                 )
          
      return stacked, ee_geom  
      
def download_tile_array(stacked_image, ee_geom, out_path, scale=TILE_RES_M):
      geemap.ee_export_image(
            stacked_image,
            filename=out_path,
            scale=scale,
            region=ee_geom,
            file_per_band=False,
            verbose=False # Suppresses the progress bar and other messages during the download process
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

# Step-6 - Main Extraction Loop

def extract_dataset(candidates, all_landslide_polys, target_count=2500, random_state=42):
      os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
      os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
      
      # Pre-converting landslide polygons to EPSG:4326 for rasterization.
      all_landslide_polys_4326 = all_landslide_polys.to_crs("EPSG:4326")
      
      #Shuffle candidates: shufflles to draw propotionally drom all the three instead of just taking whatever
      candidates = candidates.sample(frac=1, random_state=random_state).reset_index(drop=True)
      
      idx = 1
      for _, row in candidates.iterrows():
            if idx > target_count:
                  break
            try:
                  tile_geom = make_tile_geometry(row.geometry, src_crs=candidates.crs)
                  
                  center_date = row.get(ASM_DATE_FIELD)
                  if pd.isna(center_date):
                        center_date = pd.Timestamp("2017-11-01")
                  
                  result = get_stacked_image(tile_geom, center_date)
                  if result is None:
                        continue
                  stacked_image, ee_geom = result
                  
                  temp_dir = root / "output" / "temp"
                  os.makedirs(temp_dir, exist_ok=True)
                  tif_path = str(temp_dir / f"tile_{idx}.tif")
                  
                  download_tile_array(stacked_image, ee_geom, tif_path)
                  
                  with rasterio.open(tif_path) as src:
                        arr = src.read() # here the shape is (14, H, W)
                        arr = np.transpose(arr, (1, 2, 0)) # -> (H, W, 14)
                        
                  overlapping = all_landslide_polys_4326[all_landslide_polys_4326.intersects(tile_geom)]
                  mask = rasterize_mask_for_tile(overlapping.geometry.tolist(), tif_path)
                        
                  # Guard against off-by-one pixel exports -> crop or pad to exactly TILE_PX x TILE_PX
                  h, w , _ = arr.shape
                  if (h, w) != (TILE_PX, TILE_PX):
                        fixed_arr = np.zeros((TILE_PX, TILE_PX, arr.shape[2]), dtype=arr.dtype)
                        # : selects from index to the end
                        # min(h, TILE_PX) ensures that we don't go out of bounds if the image is smaller than TILE_PX
                        # min(w, TILE_PX) does the same for width
                        fixed_arr[:min(h, TILE_PX), :min(w, TILE_PX), :] = arr[:min(h, TILE_PX), :min(w, TILE_PX)]
                        arr = fixed_arr
                        
                        fixed_mask = np.zeros((TILE_PX, TILE_PX), dtype=mask.dtype)
                        fixed_mask[:min(h, TILE_PX), :min(w, TILE_PX)] = mask[:min(h, TILE_PX), :min(w, TILE_PX)]
                        mask = fixed_mask
                  
                        
                  with h5py.File(os.path.join(OUTPUT_IMG_DIR, f"image_{idx}.h5"), "w") as f:
                        f.create_dataset("img", data=arr.astype(np.float32))
                  
                  with h5py.File(os.path.join(OUTPUT_MASK_DIR, f"mask_{idx}.h5"), "w") as f:
                        f.create_dataset("mask", data=mask)
                        
                  if os.path.exists(tif_path):
                        os.remove(tif_path)
                              
                  idx += 1
                  print(f"Extracted {idx-1} / {target_count} tiles")
                  
            except Exception as e:
                  print(f"Error processing landslide {idx}: {e}")
                  if 'tif_path' in locals() and os.path.exists(tif_path):
                        os.remove(tif_path)
                  continue                 

if __name__ == "__main__":
      ee.Initialize(project=GEE_PROJECT_ID)
      
      candidates = load_inventories()
      
      extract_dataset(candidates, all_landslide_polys=candidates, target_count=2500, random_state=42)


# if __name__ == "__main__":
#       ee.Initialize(project=GEE_PROJECT_ID)
#       print(candidates.crs)
#       candidates = load_inventories()