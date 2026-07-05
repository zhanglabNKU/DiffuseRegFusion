import os
import torch
import data as Data
import model as Model
import argparse
import core.logger as Logger
import core.metrics as Metrics

import time
import numpy as np
from data.warp import Warper2d

import SimpleITK as sitk


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    image_warp = Warper2d()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_2D.json',
                        help='JSON file for configuration')
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='checkpoint prefix for validation, e.g. ./experiments/exp/checkpoint/I1000_E50')

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
    test_set = Data.create_dataset_2d_test(dataset_opt, finesize, "test")

    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    opt['path']['resume_state'] = args.weights
    # model
    diffusion = Model.create_model(opt)
    print("Model Initialized")

    registSSIM = np.zeros(len(test_set))
    originSSIM = np.zeros(len(test_set))

    NJD = np.zeros(len(test_set))
    JSD = np.zeros(len(test_set))

    registTime = []
    print('Begin Model Evaluation.')
    result_path = '{}'.format(opt['path']['results'])


    os.makedirs(result_path, exist_ok=True)
    print(len(test_loader))

    for istep, test_data in enumerate(test_loader):
        time1 = time.time()

        diffusion.feed_data(test_data)
        diffusion.test_registration()
        time2 = time.time()
        visuals = diffusion.get_current_registration()

        fusion_img = visuals["fusion_img"]
        fusion_img = (fusion_img-torch.min(fusion_img))/(torch.max(fusion_img)-torch.min(fusion_img))
        fusion_img = np.squeeze((fusion_img * 255).cpu().numpy())
        flow = visuals["flow"]

        moving = test_data['M'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        fixed = test_data['F'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        regist = image_warp(flow, moving)
        tmp_M = sitk.GetImageFromArray(moving.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_M, os.path.join(result_path, f"moving{istep}.nii.gz"))
        tmp_OM = sitk.GetImageFromArray(test_data['OM'].squeeze().cpu().numpy())
        sitk.WriteImage(tmp_OM, os.path.join(result_path, f"ori_moving{istep}.nii.gz"))
        tmp_W = sitk.GetImageFromArray(regist.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_W, os.path.join(result_path, f"regist{istep}.nii.gz"))
        tmp_F = sitk.GetImageFromArray(fixed.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_F, os.path.join(result_path, f"fixed{istep}.nii.gz"))
        flow_vis = sitk.GetImageFromArray(flow.detach().squeeze().permute(1, 2, 0).cpu().numpy())
        sitk.WriteImage(flow_vis, os.path.join(result_path, f"flow{istep}.nii.gz"))
        fusion_img = sitk.GetImageFromArray(fusion_img.squeeze())
        sitk.WriteImage(fusion_img, os.path.join(result_path, f"fusion_img{istep}.nii.gz"))



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
    print('Registration and Fusion Metrics--------------')

    print('---------------------------------------------')
    

    print('Deform Time | mean = %.3f, std= %.3f' % (mtime, stime))
    print('origin SSIM | mean = %.3f, std= %.3f' % (omssim, osssim))
    print('regist SSIM | mean = %.3f, std= %.3f' % (mssim, sssim))
    print('NJD | mean = %.3f, std= %.3f' % (mnjd, snjd))
    print('JSD | mean = %.3f, std= %.3f' % (mjsd, sjsd))
