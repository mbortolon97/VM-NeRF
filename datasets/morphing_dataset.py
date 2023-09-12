from torch.utils.data import Dataset
from datasets.cameras_combination import CamerasCombination
from complete_morph import image_morphing
from input_utils import generated_img_mask
from typing import Dict, Union
from torch import save as torch_save, cat, randint, split, full, zeros, tensor, bool as torch_bool, float32 as torch_float32, stack, Tensor, no_grad, clip, clone, no_grad, rand, ones
from torch.linalg import norm
from gc import collect as gc_collect
from utils import render_image
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from datasets.utils import Rays
from rays import get_rays
import time

from abc import ABC, abstractmethod

class AlphaDistributionGenerator(ABC):
    @abstractmethod
    def get_alpha(self, i):
        pass

class FixedAlphaGenerator(AlphaDistributionGenerator):
    def __init__(self, number_of_meta_cameras_per_interval) -> None:
        super().__init__()
        
        self.__number_of_meta_cameras_per_interval = number_of_meta_cameras_per_interval
    
    def get_alpha(self, i):
        return (float(i) + 1.) / (float(self.__number_of_meta_cameras_per_interval) + 1.)

class TruncatedNormalAlphaGenerator(AlphaDistributionGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.__normal_distribution = Normal(0.5, 0.2)

    def get_alpha(self, i):
        return clip(self.__normal_distribution.sample((1,)), 0.0, 1.0).item()

class UniformAlphaGenerator(AlphaDistributionGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.__uniform_distribution = Uniform(0.0, 1.0)

    def get_alpha(self, i):
        return self.__uniform_distribution.sample((1,)).item()

class MorphingPeriodicGeneratorDataset(Dataset):
    def __init__(
        self,
        gt_dataset,
        camera_combination_dataset : 'CamerasCombination',
        num_rays: int = None,
        near: float = None,
        far: float = None,
        number_of_meta_cameras_per_interval: int = 1,
        random_meta_camera : bool = True,
        sampling_model_type : str = 'truncated_norm',
        morph_mode : str = 'warp',
    ):
        super().__init__()

        self.gt_dataset = gt_dataset
        self.camera_combination_dataset = camera_combination_dataset
        self.color_bkgd_aug = self.gt_dataset.color_bkgd_aug

        self.num_rays = num_rays
        self.near = self.gt_dataset.NEAR if near is None else near
        self.far = self.gt_dataset.FAR if far is None else far
        self.training = True

        self.number_of_meta_cameras_per_interval = number_of_meta_cameras_per_interval
        self.alpha_generator = FixedAlphaGenerator(number_of_meta_cameras_per_interval)
        
        if random_meta_camera:
            if sampling_model_type == 'truncated_norm':
                self.alpha_generator = TruncatedNormalAlphaGenerator()
            else:
                self.alpha_generator = UniformAlphaGenerator()
        
        self.__mode = morph_mode

        assert self.__mode in ['morph','warp'], "morph mode is not valid"

        self.generate_init_camera_data()
    
    # TODO
    def generate_single_image(self, cameras_combination : 'Dict[str, Union[Dict[str, Tensor], int]]', predicted_depth : 'Tensor', alpha : 'float', device : 'str'):
        cam1_idx = cameras_combination['camera_1_idx']
        cam2_idx = cameras_combination['camera_2_idx']
        cam1_expected_results = self.gt_dataset.images[cam1_idx]
        cam2_expected_results = self.gt_dataset.images[cam2_idx]
        
        if predicted_depth is None:
            gt_shape = (*(cam1_expected_results.shape[:-1]), 1)
            
            cam1_depth = full(gt_shape, self.near, dtype=cam1_expected_results.dtype, device=device)
            cam2_depth = full(gt_shape, self.near, dtype=cam2_expected_results.dtype, device=device)
        else:
            cam1_depth = predicted_depth[cam1_idx].to(device)
            cam2_depth = predicted_depth[cam2_idx].to(device)
        
        if hasattr(self.gt_dataset, 'Ks'):
            cam1_K = self.gt_dataset.Ks[cam1_idx].to(device)
            cam2_K = self.gt_dataset.Ks[cam1_idx].to(device)
        else:
            cam1_K = self.gt_dataset.K.to(device)
            cam2_K = cam1_K
        
        cam1_c2w = self.gt_dataset.camtoworlds[cam1_idx].to(device)
        cam2_c2w = self.gt_dataset.camtoworlds[cam2_idx].to(device)
        gen_image, viewmatrix, gen_K = image_morphing(
            cam1_c2w,
            cam2_c2w,
            cam1_K,
            cam2_K,
            cam1_expected_results.to(device) / 255.0,
            cam2_expected_results.to(device) / 255.0,
            cam1_depth,
            cam2_depth,
            alpha,
            mode=self.__mode
        )

        gen_image = gen_image[0:cam1_expected_results.shape[0], 0:cam1_expected_results.shape[1], 0:cam1_expected_results.shape[2]].contiguous()
        
        if predicted_depth is None:
            gen_mask = full(cam1_expected_results.shape[:-1], False, dtype=torch_bool, device=cam1_expected_results.device)
        else:
            gen_mask = generated_img_mask(gen_image)
        
        return viewmatrix.to('cpu'), gen_image.to('cpu'), gen_K.to('cpu'), gen_mask.to('cpu')
    
    def clean_containers(self):
        self.images = None
        self.camtoworlds = None
        self.Ks = None
        self.mask = None
        self.origins = None
        self.viewdirs = None

        gc_collect()
    
    def generate_init_camera_data(self):
        with no_grad():
            self.clean_containers()

            camtoworlds = []
            images = []
            mask = []
            Ks = []

            for batch_idx in range(self.gt_dataset.camtoworlds.shape[0]):
                camtoworlds.append(self.gt_dataset.camtoworlds[batch_idx])
                images.append(self.gt_dataset.images[batch_idx] / 255.0)
                mask.append(ones(self.gt_dataset.images[batch_idx].shape[:-1], dtype=torch_bool, device=self.gt_dataset.images[batch_idx].device))
                Ks.append(self.gt_dataset.Ks[batch_idx] if hasattr(self.gt_dataset, 'Ks') else self.gt_dataset.K)
            
            for batch_idx in range(self.camera_combination_dataset.valid_combinations()):
                cameras_combination = self.camera_combination_dataset[batch_idx]

                for i in range(self.number_of_meta_cameras_per_interval):
                    alpha = self.alpha_generator.get_alpha(i)
                    
                    camtoworld, gen_image, gen_K, gen_mask = self.generate_single_image(cameras_combination, None, alpha, 'cpu')
                    camtoworlds.append(camtoworld)
                    images.append(gen_image)
                    mask.append(gen_mask)
                    Ks.append(gen_K)
            
            self.camtoworlds = stack(camtoworlds)
            self.images = stack(images)
            self.mask = stack(mask)
            self.Ks = stack(Ks)
            # torch_save({
            #     'camtoworlds': self.camtoworlds,
            #     'images': self.images,
            #     'mask': self.mask,
            #     'Ks': self.Ks
            # }, 'debug_info.pth')
            # exit(0)
            # generate the rays from the data used
            self.generate_rays()
    
    @no_grad()
    def predict_image_depth(
        self,
        radiance_field,
        occupancy_grid,
        scene_aabb : 'Tensor',
        render_step_size : 'float',
        cone_angle : 'float',
        intrinsic_parameters : 'Tensor',
        gt_image_c2w : 'Tensor',
        gt_K : 'Tensor',
        device : 'str'
    ):
        
        origins, directions = get_rays(intrinsic_parameters.to(device), gt_image_c2w[None].to(device), gt_K[None].to(device), device=device)
        viewdirs = directions / norm(
            directions, dim=-1, keepdims=True
        )
        rays = Rays(origins=origins[0], viewdirs=viewdirs[0])

        color_bkgd = ones(3, device=device)

        rgb, _, depth, _ = render_image(
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=self.near,
            far_plane=self.far,
            render_step_size=render_step_size,
            render_bkgd=color_bkgd,
            cone_angle=cone_angle,
            test_chunk_size=2048
        )
        mask = ones((rgb.shape[0], rgb.shape[1]), dtype=torch_bool)
        return rgb, depth, mask
    
    @no_grad()
    def regenerate_predict_camera_data(
        self,
        radiance_field,
        occupancy_grid,
        scene_aabb : 'Tensor',
        render_step_size : 'float',
        cone_angle : 'float',
        experiment_dir : 'str',
        device : 'str'
    ):
        self.clean_containers()

        camtoworlds = []
        images = []
        mask = []
        Ks = []

        predicted_depth = []

        radiance_field.eval()
        
        depth_start_time = time.time()
        for batch_idx in range(len(self.gt_dataset)):
            gt_c2w = self.gt_dataset.camtoworlds[batch_idx]
            gt_K = self.gt_dataset.Ks[batch_idx] if hasattr(self.gt_dataset, 'Ks') else self.gt_dataset.K
            intrinsic_parameters = tensor([self.gt_dataset.images.shape[-3], self.gt_dataset.images.shape[-2]], dtype=torch_float32, device=device)
            _, depth, gen_mask = self.predict_image_depth(radiance_field, occupancy_grid, scene_aabb, render_step_size, cone_angle, intrinsic_parameters, gt_c2w, gt_K, device)
            camtoworlds.append(gt_c2w.to('cpu'))
            mask.append(gen_mask.to('cpu'))
            images.append((self.gt_dataset.images[batch_idx]).to('cpu') / 255.0)
            Ks.append(gt_K.to('cpu'))
            predicted_depth.append(depth.to('cpu'))
        
        if False:
            from torchvision.utils import save_image
            from torch import divide
            from os.path import join
            # self.regenerate_predict_camera_data(trainer)
            pred_depth_to_save = clone(stack(predicted_depth))
            # print(pred_depth_to_save.min())
            # print(pred_depth_to_save.max())
            pred_depth_to_save = divide(pred_depth_to_save - pred_depth_to_save.min(), pred_depth_to_save.max() - pred_depth_to_save.min())
            save_image(pred_depth_to_save.permute(0,3,1,2), join(trainer.log_dir, "saved_depth.png"))
        
        generation_start_time = time.time()
        for batch_idx in range(self.camera_combination_dataset.valid_combinations()):
            cameras_combination = self.camera_combination_dataset[batch_idx]

            for i in range(self.number_of_meta_cameras_per_interval):
                alpha = self.alpha_generator.get_alpha(i)

                gen_c2w, gen_image, gen_K, gen_mask = self.generate_single_image(cameras_combination, predicted_depth, alpha, device)
                camtoworlds.append(gen_c2w)
                images.append(gen_image)
                mask.append(gen_mask)
                Ks.append(gen_K)
        
        current_time  = time.time()
        print(f"Regeneration finish, depth time {(generation_start_time - depth_start_time):.02f} s, generation time {(current_time - generation_start_time):.02f} s, total time {(current_time - depth_start_time):.02f} s")
        
        self.camtoworlds = stack(camtoworlds).contiguous()
        self.images = stack(images).contiguous()
        self.mask = stack(mask).contiguous()
        self.Ks = stack(Ks).contiguous()
        # Regenerate rays
        self.generate_rays()

        radiance_field.train()

        if True:
            from os.path import join
            self.save_gen_images(join(experiment_dir, "./generate_grid_images.png"))
    
    @no_grad()
    def save_gen_images(self, filename : 'str'):
        from torchvision.utils import save_image
        # self.regenerate_predict_camera_data(trainer)
        gen_images = clone(self.images)
        # print(gen_images.permute(0,3,1,2).shape)
        save_image(gen_images.permute(0,3,1,2), filename)

    def __len__(self) -> 'int':
        return len(self.images)
    
    def generate_rays(self):
        origins = []
        viewdirs = []
        rgba = []

        for img_idx in range(self.camtoworlds.shape[0]):
            gt_c2w = self.camtoworlds[[img_idx]]
            K_matrix = self.Ks[[img_idx]] if hasattr(self.gt_dataset, 'Ks') else self.gt_dataset.K[None]
            images_shape = self.gt_dataset.images.shape
            intrinsic_parameters = tensor([images_shape[-3], images_shape[-2]], dtype=torch_float32, device=gt_c2w.device)
            origin, viewdir = get_rays(intrinsic_parameters, gt_c2w, K_matrix, gt_c2w.device)
            mask = self.mask[img_idx]
            origins.append(origin[0, mask])
            viewdirs.append(viewdir[0, mask])
            rgba.append(self.images[img_idx, mask])
        self.origins = cat(origins, dim=0).contiguous().view(-1, 3)
        self.viewdirs = cat(viewdirs, dim=0).contiguous().view(-1, 3)
        self.rgba = cat(rgba, dim=0).contiguous().view(-1, 4)
    
    def update_num_rays(self, num_rays):
        self.num_rays = num_rays
    
    @no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data
    
    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        rays_idx = randint(0, self.origins.shape[0], size=(num_rays,), device=self.images.device)

        # generate rays
        rgba = self.rgba[rays_idx]  # (num_rays, 4)
        
        rays = Rays(origins=self.origins[rays_idx], viewdirs=self.viewdirs[rays_idx])

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }