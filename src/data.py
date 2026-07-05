import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2   

def compute_topographical_features(dem, slope_deg, res=10.0, azimuth=315, angle_altitude=45):
    """ Compute northness, eastness, profile curvature"""

    dem_padded = np.pad(dem, pad_width=1, mode="edge")

    dy, dx = np.gradient(dem_padded, res)

    d2y, _ = np.gradient(dy, res)
    _, d2x = np.gradient(dx, res)

    # removing the padding added in the beginning
    dx = dx[1:-1, 1:-1]
    dy = dy[1:-1, 1:-1]
    d2x = d2x[1:-1, 1:-1]
    d2y = d2y[1:-1, 1:-1]

    aspect = np.arctan2(-dy, dx)
    northness = np.cos(aspect)
    eastness = np.sin(aspect)

    curvature = d2x + d2y

    return northness, eastness, curvature

def compute_normalization(img_dir, file_ids): 
    """ Compute the mean and the standard deviation from the training set only. Protected against NaN vlaues. """
    N_CHANNELS = 18
    channel_sum = np.zeros(N_CHANNELS, dtype=np.float64)
    channel_squared_sum = np.zeros(N_CHANNELS, dtype=np.float64)
    pixel_count = 0
    eps = 1e-6
    
    for file_id in file_ids:
        img_path = os.path.join(img_dir, f"image_{file_id}.h5")
        if not os.path.exists(img_path):
            continue  # Skip if the file does not exist

        with h5py.File(img_path, "r") as f:
            raw_image = f["img"][:]
            
        blue  = raw_image[:, :, 1].astype(np.float32)
        green = raw_image[:, :, 2].astype(np.float32)
        red   = raw_image[:, :, 3].astype(np.float32)
        b5    = raw_image[:, :, 4].astype(np.float32)
        b6    = raw_image[:, :, 5].astype(np.float32)
        b7    = raw_image[:, :, 6].astype(np.float32)
        nir   = raw_image[:, :, 7].astype(np.float32)
        b8a   = raw_image[:, :, 8].astype(np.float32)
        swir1 = raw_image[:, :, 10].astype(np.float32)
        swir2 = raw_image[:, :, 11].astype(np.float32)
        slope = raw_image[:, :, 12].astype(np.float32)
        dem   = raw_image[:, :, 13].astype(np.float32)
        
        northness, eastness, curvature = compute_topographical_features(dem, slope)
        
        ndvi = (nir - red) / (nir + red + eps)
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
        ndwi = (green - nir) / (green + nir + eps)
        
        image_18ch = np.stack(
            [dem, slope, northness, eastness, curvature, blue, green, red, nir, b5, b6, b7, b8a, swir1, swir2, ndvi, bsi, ndwi], axis=-1
        )  # final 18 channel raster
        
        image_18ch = np.nan_to_num(image_18ch, nan=0.0)  # Replace NaN values with 0.0
        
        h, w, _ = image_18ch.shape
        
        channel_sum += np.sum(image_18ch, axis=(0, 1))
        channel_squared_sum += np.sum(image_18ch ** 2, axis=(0, 1))
        pixel_count += h * w

    mean = channel_sum / pixel_count
    std = np.sqrt((channel_squared_sum / pixel_count) - (mean ** 2))
    

    return mean.astype(np.float32), std.astype(np.float32)

def train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
            
        A.RandomRotate90(p=0.5),
            
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),
            
        ToTensorV2()
    ])
    
def val_transform():
    return A.Compose([
        ToTensorV2()
    ])

class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, file_ids=None, mean=None, std=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mean = mean
        self.std = std

        if file_ids is not None:
            self.file_ids = file_ids
        else:
            self.file_ids = sorted(
                [
                    int(f.split("_")[1].split(".")[0])
                    for f in os.listdir(img_dir)
                    if f.endswith(".h5")
                ]
            )

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        img_name = f"image_{file_id}.h5"
        mask_name = f"mask_{file_id}.h5"

        with h5py.File(os.path.join(self.img_dir, img_name), "r") as f:
            raw_image = f["img"][:]

        if self.mask_dir is not None:
            with h5py.File(os.path.join(self.mask_dir, mask_name), "r") as f:
                mask = f["mask"][:]
        else:
            mask = np.zeros((128, 128), dtype=np.int64)

        eps = 1e-6

        blue = raw_image[:, :, 1]
        green = raw_image[:, :, 2]
        red = raw_image[:, :, 3]
        b5 = raw_image[:, :, 4]
        b6 = raw_image[:, :, 5]
        b7 = raw_image[:, :, 6]
        nir = raw_image[:, :, 7]
        b8a = raw_image[:, :, 8]
        swir1 = raw_image[:, :, 10]
        swir2 = raw_image[:, :, 11]
        
        # Terrain features
        slope = raw_image[:, :, 12]
        dem = raw_image[:, :, 13]

        northness, eastness, curvature = compute_topographical_features(dem, slope)

        ndvi = (nir - red) / (nir + red + eps)

        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)

        ndwi = (green - nir) / (green + nir + eps)

        image_18ch = np.stack(
            [dem, slope, northness, eastness, curvature, blue, green, red, nir, b5, b6, b7, b8a, swir1, swir2, ndvi, bsi, ndwi], axis=-1
        ).astype(np.float32)  # final 18 channel raster
        
        if self.mean is not None and self.std is not None:
            image_18ch = (image_18ch - self.mean) / (self.std + 1e-6)

        if self.transform:
            
            augmented = self.transform(image=image_18ch, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].long()
        else:
            image_18ch = image_18ch.transpose((2, 0, 1))  # (C, H, W)

            image = torch.from_numpy(image_18ch).float()
            mask = torch.from_numpy(mask).long()
            

        return image, mask
