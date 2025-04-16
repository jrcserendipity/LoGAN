from Network_model import *
from osgeo import gdal

import argparse
import os
import torch
import time
from torchvision.utils import save_image


def data_normal(data):
    """Data normal: [min,max]--[0,1]"""
    d_min = data.min()
    d_max = data.max()
    n_data = (data - d_min) / (d_max - d_min)
    return n_data


def DTM_inference(img_name, save_path):
    """DTM inference: use the trained model generate relative height"""
    ori_tif_path = os.path.join(opt.ORIpath, img_name)
    ori_raster = gdal.Open(ori_tif_path)
    ori_raster_array = ori_raster.ReadAsArray()

    ori_nor = data_normal(ori_raster_array)
    ori_nor_tensor = torch.from_numpy(ori_nor[0]).to(torch.float32)
    ori_nor_tensor = torch.unsqueeze(ori_nor_tensor, 0)
    ori_nor_tensor = torch.unsqueeze(ori_nor_tensor, 0)
    gen_dtm = generator(ori_nor_tensor)
    filename = os.path.splitext(img_name)[0].split(".")[0]
    save_image(gen_dtm, "%s/gen_dtm_%s.tif" % (save_path, filename), gray=True, normalize=True)


if __name__ == "__main__":
    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m",
                        default=r"model/dem_reconstruction_model.pth", type=str,
                        help="Path to load model")  # load the trained model
    parser.add_argument('--ORIpath', '-i',
                        default=r"relative_height_test_sample/ORI",
                        type=str, help="Path to open ORI")  # load input ORI folder
    parser.add_argument("--result_path", "-o",
                        default=r"relative_height_test_sample/RESULT",
                        type=str,
                        help="Path to save result")  # load output relative DTM folder

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.result_path, exist_ok=True)  # new output folder

    # Setting cuda：(cuda：0)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Read ORI Files
    print("===> Read ORI Files")
    ori_lists = []  # list of prepared input ori files
    tif_file_lists = os.listdir(opt.ORIpath)
    for tif_filename in tif_file_lists:
        if os.path.splitext(tif_filename)[1].lower() == '.tif':
            ori_lists.append(tif_filename)
        ori_lists.sort(key=lambda x: int((x.split('.')[0]).rsplit("_", 1)[1]))

    # Create generator
    print("===> Building Model")
    generator = GeneratorUNet()
    # if cuda:
    #     generator = generator.cuda()

    # Load model state
    generator.load_state_dict(torch.load(opt.model))

    # Start DTM inference
    print("===> relative height inference")
    start = time.perf_counter()
    scale = 50

    for index, ori_tif_name in enumerate(ori_lists):
        DTM_inference(ori_tif_name, opt.result_path)
        i = index / len(ori_lists) * 50
        a = "*" * int(i)
        b = "." * (scale - int(i))
        c = index / len(ori_lists) * 100
        dur = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")

    print("===> Relative height inference finish")