import argparse
import os
import numpy as np
import time
import datetime
import sys

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable

from Network_model import *
from Dataset import *
from Loss_function import *

import torch

#  超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")  # 初始周期序号
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")  # 总共训练的周期
parser.add_argument("--dataset_name", type=str, default="ori_dtm", help="name of the dataset")  # 数据集的名称
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")  # 批量数据的大小
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 优化函数的学习率
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")  # 用于计算梯度的运行平均值系数
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")  # 用于计算梯度平方的运行平均值系数
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")  # 输入数据的高
parser.add_argument("--img_width", type=int, default=256, help="size of image width")  # 输入数据的宽
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 图像的通道
parser.add_argument("--sample_interval", type=int, default=500,
                    help="interval between sampling of images from generators")  # 输出图像样例之间的间隔
parser.add_argument("--checkpoint_interval", type=int, default=1,
                    help="interval between model checkpoints")  # 记录checkpoint的间隔
parser.add_argument("--train_dataset_1", "-d1",
                    default="train_dataset_256_0306.hdf5", type=str,
                    help="Path to train_Dataset1")  # 训练数据集
parser.add_argument("--train_dataset_2", "-d2",
                    default="train_dataset_128_0306.hdf5", type=str,
                    help="Path to train_Dataset2")  # 训练数据集
parser.add_argument("--val_dataset", "-vd",
                    default="validate_dataset_0306.hdf5", type=str,
                    help="Path to val_Dataset")  # 验证数据集
parser.add_argument("--test_dataset", "-td",
                    default="test_5_dataset.hdf5", type=str,
                    help="Path to test_Dataset")  # 验证数据集
opt = parser.parse_args()
print(opt)

# 新建文件地址
os.makedirs("result_images/%s" % "batches_result", exist_ok=True)  # 存储输出的结果
os.makedirs("result_images/%s" % "epoch_result", exist_ok=True)  # 存储输出的结果
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)  # 存储训练的模型
writer = SummaryWriter("train_logs")  # 记录函数图像

# 设置cuda：（cuda：0）
cuda = True if torch.cuda.is_available() else False

# 损失函数 Loss functions
criterion_GAN = torch.nn.MSELoss()  # GAN_loss
criterion_pixelwise = torch.nn.L1Loss()  # L1loss

criterion_bh_loss = BerhuLoss()  # BerhuLoss
criterion_gr_loss = GradientLoss()  # GradientLoss

# L1损失函数的权重 Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 10
# lamdba_bh = 10
lamdba_gr = 100

# 判别器Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# 创建生成器和判别器对象：Initialize generator and discriminator
print("===> Building Model")
generator = GeneratorUNet()
discriminator = Discriminator()

# 如果有显卡在cuda模式中运行
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criterion_bh_loss.cuda()
    criterion_gr_loss.cuda()

# 是否有已经运行好的模型
if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(
        torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))  # 加载生成网络模型
    discriminator.load_state_dict(
        torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))  # 加载判别网络模型
else:
    # 权重初始化 Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# 定义优化函数 Optimizers
print("===> Setting Optimizer")
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 加载数据
print("===> Loading datasets")

train_set1 = DEMDataset(opt.train_dataset_1)  # 读取h5fd训练数据集1
dataloader1 = DataLoader(dataset=train_set1, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True, drop_last=True)  # 处理输入数据

train_set2 = DEMDataset(opt.train_dataset_2)  # 读取h5fd训练数据集1
dataloader2 = DataLoader(dataset=train_set2, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True, drop_last=True)  # 处理输入数据

val_set = DEMDataset(opt.val_dataset)  # 读取h5fd验证数据集
val_dataloader = DataLoader(dataset=val_set, num_workers=1, batch_size=opt.batch_size, shuffle=True, drop_last=True)  # 处理输入数据

test_set = DEMDataset(opt.test_dataset)  # 读取h5fd数据集
test_dataloader = DataLoader(dataset=test_set, num_workers=1, batch_size=5,
                             shuffle=True, drop_last=True)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def data_normal(data):
    d_min = torch.min(data)
    d_max = torch.max(data)
    n_data = (data - d_min) / (d_max - d_min)
    return n_data



def sample_images(batches_done, style):
    """Saves a generated sample from dataset set"""
    imgs = next(iter(test_dataloader))
    real_ori = Variable(imgs["ori"].type(Tensor))
    real_dtm = Variable(imgs["dtm"].type(Tensor))

    real_ori_nor = data_normal(real_ori)
    real_dtm_nor = data_normal(real_dtm)

    real_ori_nor = torch.unsqueeze(real_ori_nor, 1)
    real_dtm_nor = torch.unsqueeze(real_dtm_nor, 1)

    fake_dtm = generator(real_ori_nor)

    img_sample = torch.cat((real_ori_nor.data, fake_dtm.data, real_dtm_nor.data), -2)
    save_image(img_sample, "result_images/%s/all_%s.png" % (style, batches_done), nrow=5, normalize=True)


# 每个epoch进行验证
def validate(e):
    val_imgs = next(iter(val_dataloader))
    val_real_dtm = Variable(val_imgs["dtm"].type(Tensor))  # （1，256，256）
    val_real_ori = Variable(val_imgs["ori"].type(Tensor))  # （1，256，256）

    # 数据归一化处理 inputs normalize
    val_real_ori = data_normal(val_real_ori)
    val_real_dtm = data_normal(val_real_dtm)
    val_real_dtm = torch.unsqueeze(val_real_dtm, 1)  # 升维函数:(1, 1, 256, 256)
    val_real_ori = torch.unsqueeze(val_real_ori, 1)  # 升维函数:(1, 1, 256, 256)

    # 对抗性地面真值 Adversarial ground truths
    patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)
    val_valid = Variable(Tensor(np.ones((val_real_ori.size(0), *patch))),
                         requires_grad=False)  # 定义真的图片的label为1.0  valid：（1，1，16，16）
    val_fake = Variable(Tensor(np.zeros((val_real_ori.size(0), *patch))),
                        requires_grad=False)  # 定义假的图片的label为0.0  valid：（1，1，16，16）
    # GAN loss
    val_fake_dtm = generator(val_real_ori)  # 用生成器生成图片（1，1，256，256）
    val_pred_fake = discriminator(val_fake_dtm, val_real_ori)  # patchGAN：用判别器判定后得到标签（1，1，16，16）
    val_loss_GAN = criterion_GAN(val_pred_fake, val_valid)

    val_loss_bh = criterion_bh_loss(val_real_dtm, val_fake_dtm)
    val_loss_pixel = criterion_pixelwise(fake_dtm, real_dtm)
    val_loss_gr = criterion_gr_loss(val_real_dtm, val_fake_dtm)

    # rms
    val_fake_dtm_n = val_fake_dtm.cpu().detach().numpy()
    val_real_dtm_n = val_real_dtm.cpu().detach().numpy()
    val_mse = (np.abs(val_fake_dtm_n - val_real_dtm_n) ** 2)
    val_rmse = np.sqrt(val_mse).mean()

    # Real loss： 真实图片与真值label之间的loss
    val_pred_real = discriminator(val_real_dtm, val_real_ori)
    val_loss_real = criterion_GAN(val_pred_real, val_valid)

    # Fake loss： 生成图片与假值label之间的loss
    val_pred_fake = discriminator(val_fake_dtm.detach(), val_real_ori)
    val_loss_fake = criterion_GAN(val_pred_fake, val_fake)

    # Total loss
    val_loss_G = val_loss_GAN + lambda_pixel * val_loss_pixel + lamdba_gr * val_loss_gr
    val_loss_D = 0.5 * (val_loss_real + val_loss_fake)  # 判别器的损失为判真损失与判假损失的均值

    # 记录训练函数
    writer.add_scalar('val_gen_loss', val_loss_G, e)
    writer.add_scalar('val_dis_loss', val_loss_D, e)
    writer.add_scalar('val_rmse', val_rmse, e)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    if epoch % 2 == 0:
        dataloader = dataloader1
        patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)
    else:
        dataloader = dataloader2
        patch = (1, 128 // 2 ** 4, 128 // 2 ** 4)
    for i, batch in enumerate(dataloader):
        # 模型输入 Model inputs
        real_dtm = Variable(batch["dtm"].type(Tensor))  # （1，256，256）
        real_ori = Variable(batch["ori"].type(Tensor))  # （1，256，256）

        # 数据归一化处理 inputs normalize
        real_ori = data_normal(real_ori)
        real_dtm = data_normal(real_dtm)
        real_dtm = torch.unsqueeze(real_dtm, 1)  # 升维函数:(1, 1, 256, 256)
        real_ori = torch.unsqueeze(real_ori, 1)  # 升维函数:(1, 1, 256, 256)

        # 对抗性地面真值 Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_ori.size(0), *patch))),
                         requires_grad=False)  # 定义真的图片的label为1.0  valid：（1，1，16，16）
        fake = Variable(Tensor(np.zeros((real_ori.size(0), *patch))),
                        requires_grad=False)  # 定义假的图片的label为0.0  valid：（1，1，16，16）

        # ------------------
        #  训练生成器 Train Generators
        # ------------------

        optimizer_G.zero_grad()  # 反向传播前，先梯度归零

        # GAN loss
        fake_dtm = generator(real_ori)  # 用生成器生成图片（1，1，256，256）
        pred_fake = discriminator(fake_dtm, real_ori)  # patchGAN：用判别器判定后得到标签（1，1，16，16）
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_dtm, real_dtm)
        # loss_bh = criterion_bh_loss(real_dtm, fake_dtm)
        loss_gr = criterion_gr_loss(real_dtm, fake_dtm)

        # rms
        fake_dtm_n = fake_dtm.cpu().detach().numpy()
        real_dtm_n = real_dtm.cpu().detach().numpy()
        mse = (np.abs(fake_dtm_n - real_dtm_n) ** 2)
        rmse = np.sqrt(mse).mean()

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel + lamdba_gr * loss_gr  # lambda = 100
        # loss_G = loss_GAN + lamdba_bh * loss_bh + lamdba_gr * loss_gr  # lambbh = 10 gr=100
        loss_G.backward()  # 误差反向传播

        optimizer_G.step()  # 更新生成网络参数

        # ---------------------
        #  训练判别器 Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()  # 反向传播前，先梯度归零

        # Real loss： 真实图片与真值label之间的loss
        pred_real = discriminator(real_dtm, real_ori)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss： 生成图片与假值label之间的loss
        pred_fake = discriminator(fake_dtm.detach(), real_ori)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)  # 判别器的损失为判真损失与判假损失的均值

        loss_D.backward()  # 误差反向传播
        optimizer_D.step()  # 更新判别网络参数

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # 打印训练过程的日志 Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, loss_gh: %f, loss_bh: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_gr.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # 保存损失函数图像 Save loss function
        if i == len(dataloader) - 1:
            writer.add_scalar('train_gen_loss', loss_G, epoch)
            writer.add_scalar('train_dis_loss', loss_D, epoch)
            writer.add_scalar('train_gen_depth_loss', loss_pixel, epoch)
            writer.add_scalar('train_gen_grad_loss', loss_gr, epoch)
            writer.add_scalar('train_gen_GAN_loss', loss_GAN, epoch)
            writer.add_scalar('train_rmse', rmse, epoch)
            sample_images(epoch, "epoch_result")

        # 保存训练过程的图像：If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, "batches_result")

    # 验证集的情况
    print("validation")
    validate(epoch)
    print("validation finsh")

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 保存训练模型：Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
