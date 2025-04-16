import os
import cv2
import time
import numpy as np
from osgeo import gdal


def arr2raster(arr, raster_file, prj=None, trans=None):
    """   Converting arrays to rasters   """
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    dst_ds.GetRasterBand(1).WriteArray(arr)

    dst_ds.FlushCache()


def is_number(s):
    """   Determine whether a string is a number   """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)  #
        return True
    except (TypeError, ValueError):
        pass


def file_create(filepath):
    """   Create a folder   """
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def expand_tif_list(file_path, tif_lists):
    """   Extract tif file lists from folders   """
    tif_file_lists = os.listdir(file_path)
    for tif_filename in tif_file_lists:
        if os.path.splitext(tif_filename)[1].lower() == '.tif':
            filename = os.path.splitext(tif_filename)[0].split(".")[0]
            Namenum = filename.rsplit("_")[1]
            if is_number(Namenum):
                tif_lists.append(tif_filename)
    tif_lists.sort(key=lambda x: int((x.split('.')[0]).rsplit("_")[1]))


if __name__ == "__main__":
    # ---------------------------------  File path ---------------------------------
    print("===> load file path")
    # Input folders filepath: high-resolution image, relative dtm, low-resolution
    input_img_filepath = r"E:\01-LoGAN\03_Reconstruction_result\001462_2015\new_result\ori_tiles"
    gen_dtm_filepath = r"E:\01-LoGAN\03_Reconstruction_result\001462_2015\new_result\gen_dtm"
    low_dtm_filepath = r"E:\01-LoGAN\03_Reconstruction_result\001462_2015\new_result\low_dtm_tiles"
    input_img_lists = []
    gen_dtm_lists = []
    low_dtm_lists = []

    # Output folders filepath
    gendem_abs_add_lowpass_filepath = r"E:\01-LoGAN\03_Reconstruction_result\001462_2015\new_result\abs_output_tiles"
    file_create(gendem_abs_add_lowpass_filepath)

    # Extract tif file lists from folders
    expand_tif_list(input_img_filepath, input_img_lists)
    expand_tif_list(gen_dtm_filepath, gen_dtm_lists)
    expand_tif_list(low_dtm_filepath, low_dtm_lists)

    # ---------------------------------  Post process ---------------------------------
    print("===> postprocess")

    start = time.perf_counter()
    scale = 50

    for index, gen_dtm_tif_name in enumerate(gen_dtm_lists):
        # read tif name
        low_dtm_tif_name = low_dtm_lists[index]
        input_img_tif_name = input_img_lists[index]

        # 0) Open raster files
        input_img_tif_path = os.path.join(input_img_filepath, input_img_tif_name)
        gen_dem_tif_path = os.path.join(gen_dtm_filepath, gen_dtm_tif_name)
        low_dem_tif_path = os.path.join(low_dtm_filepath, low_dtm_tif_name)
        gen_dem_add_different_map_lowpass_path = os.path.join(gendem_abs_add_lowpass_filepath, gen_dtm_tif_name)

        # Open gen_dtm raster
        gen_dem_raster = gdal.Open(gen_dem_tif_path)
        gen_dem_raster_Xsize = int(gen_dem_raster.RasterXSize)
        gen_dem_raster_Ysize = int(gen_dem_raster.RasterYSize)
        gen_dem_raster_array = \
        gen_dem_raster.ReadAsArray(buf_xsize=gen_dem_raster_Xsize, buf_ysize=gen_dem_raster_Ysize)
        if np.min(gen_dem_raster_array[0]) <= 1e-8:
            gen_dem_raster_array = np.pad(gen_dem_raster_array[1:], pad_width=((1, 0), (0, 0)), mode='edge')

        # Open low_dtm raster
        low_dem_raster = gdal.Open(low_dem_tif_path)
        low_dem_raster_Xsize = int(low_dem_raster.RasterXSize)
        low_dem_raster_Ysize = int(low_dem_raster.RasterYSize)
        low_dem_raster_array \
            = low_dem_raster.ReadAsArray(buf_xsize=low_dem_raster_Xsize,
                                         buf_ysize=low_dem_raster_Ysize)
        if np.min(low_dem_raster_array[0]) <= -32768:
            low_dem_raster_array =  np.pad(low_dem_raster_array[1:], pad_width=((1, 0), (0, 0)), mode='edge')

        # Open input_img raster
        input_img_raster = gdal.Open(input_img_tif_path)
        input_img_raster_Xsize = int(input_img_raster.RasterXSize)
        input_img_raster_Ysize = int(input_img_raster.RasterYSize)

        # 1) get mean and std from low_DEM
        low_dem_mean = np.mean(low_dem_raster_array)
        low_dem_std = np.std(low_dem_raster_array)

        # 2) stretch relative heights to absolute scales
        gen_dtm_nor_array = (gen_dem_raster_array - np.mean(gen_dem_raster_array)) / np.std(gen_dem_raster_array)  # 归一化
        gen_abs_dem_array = np.array(gen_dtm_nor_array * low_dem_std + low_dem_mean, dtype=float)

        # 3）add low-frequency terrain trends
        gen_abs_dem_array_downsample = cv2.resize(gen_abs_dem_array,
                                                  dsize=(low_dem_raster_Xsize, low_dem_raster_Ysize),
                                                  interpolation=cv2.INTER_LINEAR)
        low_dem_raster_array_blur = cv2.GaussianBlur(low_dem_raster_array, (17, 17), 1.3)
        gen_abs_dem_array_blur = cv2.GaussianBlur(gen_abs_dem_array_downsample, (17, 17), 1.3)
        different_map_array = np.array(low_dem_raster_array_blur - gen_abs_dem_array_blur, dtype=float)
        different_map_lowpass_array = cv2.GaussianBlur(different_map_array, (13, 13), 1.3)
        different_map_lowpass_array_upsample = cv2.resize(different_map_lowpass_array,
                                                          dsize=(gen_dem_raster_Xsize, gen_dem_raster_Ysize),
                                                          interpolation=cv2.INTER_LINEAR)
        gen_dem_add_different_map_lowpass_array = np.array(gen_abs_dem_array + different_map_lowpass_array_upsample,
                                                           dtype=float)

        # 4）add projection and transform information
        input_img_projection = input_img_raster.GetProjection()
        input_img_transform = input_img_raster.GetGeoTransform()

        i = index / len(gen_dtm_lists) * 50
        a = "*" * int(i)
        b = "." * (scale - int(i))
        c = index / len(gen_dtm_lists) * 100
        dur = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")

        # save DTM results
        arr2raster(gen_dem_add_different_map_lowpass_array, gen_dem_add_different_map_lowpass_path,
                   prj=input_img_projection, trans=input_img_transform)
