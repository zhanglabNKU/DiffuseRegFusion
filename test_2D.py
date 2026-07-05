import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from math import *
import time
import numpy as np
import torch.nn.functional as F
from data.warp import Warper2d
from model.diffusion_3D.unet import SpatialTransform
import SimpleITK as sitk

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    image_warp = Warper2d()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_2D.json',
                        help='JSON file for configuration')
    parser.add_argument('-w', '--weights', type=str,
                        default='./experiments/swin_unetR_aug+1.0sim+0.1reg@adamw[1e-4,1e-4]cosine[1e-6]_train_250908_121240/checkpoint/I192000_E1200',
                        help='weights file for validation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    phase = 'test'
    finesize = opt['model']['diffusion']['image_size']
    dataset_opt = opt['datasets']['test']
    test_set = Data.create_dataset_ACDC_test(dataset_opt, finesize, "test")
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    opt['path']['resume_state'] = args.weights
    # model
    diffusion = Model.create_model(opt)
    stn = SpatialTransform(finesize).cuda()  # stn的作用是根据输入的变形场进行变换，本身不具有可以学习的参数
    print("Model Initialized")
    # Train

    registSSIM = np.zeros(len(test_set))
    originSSIM = np.zeros(len(test_set))
    flowSSIM = np.zeros(len(test_set))
    NJD = np.zeros(len(test_set))
    JSD = np.zeros(len(test_set))

    registTime = []
    print('Begin Model Evaluation.')
    idx_ = 0
    result_path = '{}'.format(opt['path']['results'])

    os.makedirs(result_path, exist_ok=True)
    print(len(test_loader))

    for istep, test_data in enumerate(test_loader):
        idx_ += 1
        dataName = istep
        time1 = time.time()
        diffusion.feed_data(test_data)
        diffusion.test_registration()
        time2 = time.time()
        process_time = time2 - time1  # 定义处理时间变量
        visuals = diffusion.get_current_registration()
        fusion_img = visuals["fusion_img"]
        fusion_img = (fusion_img-torch.min(fusion_img))/(torch.max(fusion_img)-torch.min(fusion_img))
        fusion_img = np.squeeze((fusion_img * 255).cpu().numpy())
        flow = visuals["flow"]

        moving = test_data['M'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        fixed = test_data['F'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        regist = image_warp(flow, moving)
        # flow_truth = test_data['flow'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        # regist = stn(moving.type(torch.float32), flow)

        # save Intermediate results of the sampling process
        # for idx, item in enumerate(visuals['contF']):
        #     flow_vis = sitk.GetImageFromArray(item.detach().squeeze().permute(1, 2, 0).cpu().numpy())
        #     sitk.WriteImage(flow_vis, f"./toy_sample_2D_new/x_start{istep}_{idx}.nii.gz")
        #     regist_tmp = stn(moving.type(torch.float32), item)
        #     regist_tmp_vis = sitk.GetImageFromArray(regist_tmp.squeeze().cpu().numpy())
        #     sitk.WriteImage(regist_tmp_vis, f"./toy_sample_2D_new/regist{istep}_{idx}.nii.gz")

        tmp_M = sitk.GetImageFromArray(moving.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_M, f"./results_nii/20/moving{istep}.nii.gz")
        tmp_OM = sitk.GetImageFromArray(test_data['OM'].squeeze().cpu().numpy())
        sitk.WriteImage(tmp_OM, f"./results_nii/20/ori_moving{istep}.nii.gz")
        tmp_W = sitk.GetImageFromArray(regist.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_W, f"./results_nii/20/regist{istep}.nii.gz")
        tmp_F = sitk.GetImageFromArray(fixed.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_F, f"./results_nii/20/fixed{istep}.nii.gz")
        flow_vis = sitk.GetImageFromArray(flow.detach().squeeze().permute(1, 2, 0).cpu().numpy())
        sitk.WriteImage(flow_vis, f"./results_nii/20/flow{istep}.nii.gz")
        # fusion_img = sitk.GetImageFromArray(fusion_img.squeeze().cpu().numpy())
        fusion_img = sitk.GetImageFromArray(fusion_img.squeeze())
        sitk.WriteImage(fusion_img, f"./results_nii/20/fusion_img{istep}.nii.gz")
        # tmp_FT = sitk.GetImageFromArray(flow_truth.squeeze().cpu().numpy())
        # sitk.WriteImage(tmp_FT, f"./toy_sample_2D_new/flow_truth{istep}.nii.gz")


        ssim_regist = round(diffusion.netG.loss_ssim(regist, test_data['OM']).item(), 4)
        ssim_origin = round(diffusion.netG.loss_ssim(moving, test_data['OM']).item(), 4)
        njd_flow = Metrics.neg_jacobian_det(flow).item()
        jsd_flow = Metrics.jacobian_det_var(flow).item()

        registSSIM[istep] = ssim_regist
        originSSIM[istep] = ssim_origin
        NJD[istep] = njd_flow
        JSD[istep] = jsd_flow
        print('---- Original SSIM: %03f | Deformed SSIM: %03f' % (ssim_origin, ssim_regist))
        print('---- NJD: %03f | JSD: %03f' % (njd_flow, jsd_flow))


        registTime.append(time2 - time1)
        time.sleep(1)

    mtime, stime = np.mean(registTime), np.std(registTime)
    omssim, osssim = np.mean(originSSIM), np.std(originSSIM)
    mssim, sssim = np.mean(registSSIM), np.std(registSSIM)
    mnjd, snjd = np.mean(NJD), np.std(NJD)
    mjsd, sjsd = np.mean(JSD), np.std(JSD)

    print()
    print('---------------------------------------------')
    print('Total Dice and Time Metrics------------------')
    print('---------------------------------------------')
    

    print('Deform Time | mean = %.3f, std= %.3f' % (mtime, stime))
    print('origin SSIM | mean = %.3f, std= %.3f' % (omssim, osssim))
    print('regist SSIM | mean = %.3f, std= %.3f' % (mssim, sssim))
    print('NJD | mean = %.3f, std= %.3f' % (mnjd, snjd))
    print('JSD | mean = %.3f, std= %.3f' % (mjsd, sjsd))
