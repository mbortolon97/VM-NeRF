"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.mlp import VanillaNeRFRadianceField
from utils import render_image, set_random_seed
from scenes_list import scenes_list
from cv2 import VideoWriter_fourcc, VideoWriter
from datasets.utils import Rays
from icecream import ic

from nerfacc import ContractionType, OccupancyGrid

def split_aabb(s):
    print("aabb: ", s)
    return [float(item) for item in s.split(",")]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="data root",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="chair",
        choices=scenes_list.keys(),
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=float,
        nargs=6,
        default=[-1.5,-1.5,-1.5,1.5,1.5,1.5],
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument(
        "--vm",
        action="store_true",
        help="whether to use view morphing augmentation technique",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="whether to execute an overfit",
    )
    parser.add_argument(
        "--vm_change_every",
        type=int,
        default=500,
        help="after how much iteration change novel views",
    )
    parser.add_argument(
        "--vm_warmup",
        type=int,
        default=500,
        help="warmup number of iterations",
    )
    parser.add_argument(
        "--vm_max_distance",
        type=float,
        default=5.0,
        help="the maximum distance between cameras",
    )
    parser.add_argument(
        "--vm_morph_mode",
        type=str,
        choices=['morph','warp'],
        default="warp",
        help="the type of morph to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="where to save experiments",
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    return args

def setup_dataset(args, device):
    render_n_samples = 1024

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    morphing_dataset_kwargs = {"morph_mode": args.vm_morph_mode}
    scene_type = scenes_list[args.scene]
    if scene_type == "mipnerf360":
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = args.data
        target_sample_batch_size = 1 << 16
        # target_sample_batch_size = 1 << 14
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 128
    elif scene_type == "nerf_synthetic":
        from datasets.nerf_synthetic import SubjectLoader

        data_root_fp = args.data
        target_sample_batch_size = 1 << 16
        # target_sample_batch_size = 1 << 14
        grid_resolution = 128
    elif scene_type == "dtu":
        from datasets.dtu import SubjectLoader

        data_root_fp = args.data
        target_sample_batch_size = 1 << 16
        grid_resolution = 128

        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4,"n_input_views":3}
        test_dataset_kwargs = {"factor": 4}

        pass

    if args.vm:
        train_dataset_kwargs['batch_over_images'] = False

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    if args.vm:
        from datasets.morphing_dataset import MorphingPeriodicGeneratorDataset
        from datasets.cameras_combination import CamerasCombination

        cameras_combination_dataset = CamerasCombination(
            train_dataset,
            args.vm_max_distance,
        )
        gt_dataset = train_dataset
        train_dataset = MorphingPeriodicGeneratorDataset(
            gt_dataset,
            cameras_combination_dataset,
            num_rays=target_sample_batch_size // render_n_samples,
            **morphing_dataset_kwargs,
        )
    
    # train_dataset.images = train_dataset.images.to(device)
    # train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    # train_dataset.K = train_dataset.K.to(device)

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )
    # test_dataset.images = test_dataset.images.to(device)
    # test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    # test_dataset.K = test_dataset.K.to(device)
    if args.overfit == True:
        test_dataset = train_dataset

    # learn output dir
    experiment_dir = os.path.join(args.output_dir, scene_type, f"{args.scene}_{args.vm_morph_mode}")
    os.makedirs(experiment_dir, exist_ok=True)

    return (target_sample_batch_size, contraction_type, scene_aabb, near_plane, far_plane, render_step_size, grid_resolution, train_dataset, test_dataset, experiment_dir)

def eval(radiance_field, occupancy_grid, args, test_dataset, scene_aabb, near_plane, far_plane, render_step_size, experiment_dir, device):
    # evaluation
    radiance_field.eval()

    rgb_fourcc = VideoWriter_fourcc(*'mp4v')
    rgb_writer = None
    rgb_tmp_video_path = os.path.join(experiment_dir, 'tmp_rgb_test_render.mp4')
    rgb_result_dir_path = os.path.join(experiment_dir, 'result')
    os.makedirs(rgb_result_dir_path, exist_ok=True)

    # all_origins = []
    # all_viewdirs = []

    psnrs = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_dataset))):
            data = test_dataset[i]
            render_bkgd = data["color_bkgd"].to(device)

            # all_origins.append(data["rays"].origins)
            # all_viewdirs.append(data["rays"].viewdirs)

            rays = Rays(origins=data["rays"].origins.to(device), viewdirs=data["rays"].viewdirs.to(device))
            pixels = data["pixels"].to(device)

            # rendering
            rgb, acc, depth, _ = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                # test options
                test_chunk_size=args.test_chunk_size,
            )

            # render

            if len(rgb.shape) == 4:
                rgb = rgb[0]
            if len(pixels.shape) == 4:
                pixels = pixels[0]
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            if rgb_writer is None:
                rgb_writer = VideoWriter(rgb_tmp_video_path, rgb_fourcc, 10.0, tuple(np.asarray(rgb.shape)[[1,0]]))
            rgb_writer.write((rgb.cpu().numpy() * 255).astype(np.uint8)[..., [2,1,0]])
            # imageio.imwrite(
            #     "acc_binary_test.png",
            #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
            # )
            imageio.imwrite(
                os.path.join(rgb_result_dir_path, f"{i:03d}.png"),
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            # break
        if rgb_writer is not None:
            rgb_writer.release()
    
    # all_origins = torch.stack(all_origins).view(-1, 3)
    # all_viewdirs = torch.stack(all_viewdirs).view(-1, 3)
    # torch.save({
    #     'origins': all_origins[::743],
    #     'viewdirs': all_viewdirs[::743],
    # }, 'rays_morphing.pth')

    psnr_avg = sum(psnrs) / len(psnrs)
    print(f"evaluation: psnr_avg={psnr_avg}")

def train():
    device = "cuda:0"
    set_random_seed(42)

    args = parse_args()

    target_sample_batch_size, contraction_type, scene_aabb, near_plane, far_plane, render_step_size, grid_resolution, train_dataset, test_dataset, experiment_dir = setup_dataset(args, device)

    # setup the radiance field we want to train.
    max_steps = 50000
    eval_steps = 5000
    grad_scaler = torch.cuda.amp.GradScaler(1)
    radiance_field = VanillaNeRFRadianceField().to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            max_steps // 2,
            max_steps * 3 // 4,
            max_steps * 5 // 6,
            max_steps * 9 // 10,
        ],
        gamma=0.33,
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    
    # training
    step = 0
    tic = time.time()
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"].to(device)
            rays = Rays(origins=data["rays"].origins.to(device), viewdirs=data["rays"].viewdirs.to(device))
            pixels = data["pixels"].to(device)

            # update occupancy grid
            occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: radiance_field.query_opacity(
                    x, render_step_size
                ),
            )

            # render
            # scene_aabb
            rgb, acc, depth, n_rendering_samples = render_image(
                radiance_field,
                None,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
            )
            if n_rendering_samples == 0:
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            if step % 1000 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | "
                    f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                )
                torch.save({'epoch': epoch, 'radiance_field': radiance_field.state_dict(),'occupancy_grid':occupancy_grid.state_dict(),'loss':loss.item(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}, os.path.join(experiment_dir, 'checkpoint.ckpt'))

            if step >= 0 and step % eval_steps == 0 and step > 0:
                test_dataset.training = False
                train_dataset.training = False
                # occupancy_grid
                eval(radiance_field, None, args, test_dataset, scene_aabb, near_plane, far_plane, render_step_size, experiment_dir, device)
                test_dataset.training = True
                train_dataset.training = True

            if step >= max_steps:
                print("training stops")
                exit()
            
            if args.vm and step % args.vm_change_every == 0 and step >= args.vm_warmup:
                # occupancy_grid
                train_dataset.regenerate_predict_camera_data(
                    radiance_field,
                    None,
                    scene_aabb,
                    render_step_size,
                    args.cone_angle,
                    experiment_dir,
                    device
                )

            step += 1

    

if __name__ == "__main__":
    train()
    
    

    
