import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


def compute_topographical_features(
    dem, slope_deg, res=10.0, azimuth=315, angle_altitude=45
):

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

    curvature = d2x + d2y

    slope_rad = np.radians(slope_deg)
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(angle_altitude)

    hillshade = np.sin(altitude_rad) * np.sin(slope_rad) + np.cos(
        altitude_rad
    ) * np.cos(azimuth_rad - aspect)
    hillshade = np.clip(hillshade, 0, 1)

    return northness, curvature, hillshade


class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

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
        nir = raw_image[:, :, 7]
        swir1 = raw_image[:, :, 10]
        slope = raw_image[:, :, 12]
        dem = raw_image[:, :, 13]

        northness, curvature, hillshade = compute_topographical_features(dem, slope)

        ndvi = (nir - red) / (nir + red + eps)

        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)

        ndwi = (green - nir) / (green + nir + eps)

        image_8ch = np.stack(
            [dem, slope, northness, curvature, hillshade, ndvi, bsi, ndwi], axis=-1
        )  # final 8 channel raster

        image_8ch = image_8ch.transpose((2, 0, 1))  # (C, H, W)

        image = torch.from_numpy(image_8ch).float()
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
