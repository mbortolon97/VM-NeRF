from numpy import meshgrid
from torch import Tensor, ones_like, cat, flatten, ceil, floor, multiply, divide, tensor, int32 as torch_int, int64 as torch_long, max, meshgrid, stack, arange, flip
from torch.nn.functional import unfold
from rays import get_rays, ndc_rays
from typing import Tuple

def transform_in_rays(intrinsic_parameter : 'Tensor', viewmatrix : 'Tensor', K : 'Tensor', near_bounds : 'Tensor', far_bounds : 'Tensor', device : 'str', ndc : 'bool') -> 'Tuple[Tensor,Tensor,Tensor,Tensor]':
    ray_ori, ray_dir = get_rays(intrinsic_parameter, viewmatrix, K, device)

    if ndc:
        ray_ori, ray_dir = ndc_rays(intrinsic_parameter, 1., ray_ori, ray_dir)
    
    near_arr = near_bounds * ones_like(ray_ori[..., :1])
    far_arr = far_bounds * ones_like(ray_ori[..., :1])

    return ray_ori, ray_dir, near_arr, far_arr

def correct_background_type(new_bkg : 'str', white_bkg : 'bool') -> 'str':
    if new_bkg == '':
        new_bkg = 'white' if white_bkg else 'black'
    return new_bkg

def get_input_data(
        viewmatrix: 'Tensor',
        intrinsic_parameter: 'Tensor',
        near_bounds: 'Tensor',
        far_bounds: 'Tensor',
        K: 'Tensor',
        ndc: 'bool',
        device : 'str') -> 'Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]':
    rays_ori = []
    rays_dir = []
    near_arr = []
    far_arr = []

    for i in range(viewmatrix.shape[0]):
        ray_ori, ray_dir, near_arr_img, far_arr_img = transform_in_rays(intrinsic_parameter[i], viewmatrix[[i]], K[[i]], near_bounds[i], far_bounds[i], device, ndc)

        rays_ori += [ray_ori]
        rays_dir += [ray_dir]

        near_arr += [near_arr_img]
        far_arr += [far_arr_img]

    rays_ori = cat(rays_ori, 0).view(-1, 3)
    rays_dir = cat(rays_dir, 0).view(-1, 3)
    near_arr = cat(near_arr, 0).view(-1, 1)
    far_arr = cat(far_arr, 0).view(-1, 1)
    
    return rays_ori, rays_dir, near_arr, far_arr

def flatten_vectors_prediction(rays_dir : 'Tensor', rays_ori : 'Tensor', bkg_color : 'Tensor', near_arr : 'Tensor', far_arr : 'Tensor', viewdirs : 'Tensor') -> 'Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]':
    rays_dir = flatten(rays_dir, start_dim=0, end_dim=-2)
    rays_ori = flatten(rays_ori, start_dim=0, end_dim=-2)
    bkg_color = flatten(bkg_color, start_dim=0, end_dim=-2)
    near_arr = flatten(near_arr, start_dim=0, end_dim=-2)
    far_arr = flatten(far_arr, start_dim=0, end_dim=-2)
    viewdirs = flatten(viewdirs, start_dim=0, end_dim=-2)

    return rays_dir, rays_ori, bkg_color, near_arr, far_arr, viewdirs

def flatten_vectors(rays_dir : 'Tensor', rays_ori : 'Tensor', expect_results : 'Tensor', bkg_color: 'Tensor', near_arr : 'Tensor', far_arr : 'Tensor', viewdirs : 'Tensor') -> 'Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]':
    rays_dir, rays_ori, bkg_color, near_arr, far_arr, viewdirs = flatten_vectors_prediction(rays_dir, rays_ori, bkg_color, near_arr, far_arr, viewdirs)
    expect_results = flatten(expect_results, start_dim=0, end_dim=-2)

    return rays_dir, rays_ori, expect_results, bkg_color, near_arr, far_arr, viewdirs

def resize_depthmap(depthmap : 'Tensor', dest_size : 'Tensor', use_median=False):
    original_size = tensor(depthmap.size(), dtype=depthmap.dtype)
    resizing_ratios = divide(original_size, dest_size)

    stride = floor(resizing_ratios)

    kernel_size = ceil(resizing_ratios)

    padding = divide(multiply(ceil(divide(original_size, stride)), stride) + (kernel_size - stride) - original_size, 2.)

    fold_params = dict(kernel_size=tuple(kernel_size.type(torch_int).tolist()), dilation=1, padding=tuple(padding.type(torch_int).tolist()), stride=tuple(stride.type(torch_int).tolist()))
    patched = unfold(depthmap[None, None, ...], **fold_params)

    if use_median:
        depth_result = patched.median(dim=1, keepdim=True).values
    else:
        depth_result = patched.cpu().mode(dim=1, keepdim=True).values.to(depthmap.device)
    

    destination_depth = tuple(dest_size.type(torch_long).tolist())

    return depth_result.view(*destination_depth)

def generated_img_mask(img):
    mask = img[..., -1] > 0.5
    mask_max_values_lr, mask_max_indices_lr = max(mask, dim=1, keepdim=True)
    mask_max_indices_lr[mask_max_values_lr == 0] = mask.size(dim=1) - 1
    mask_max_values_rl, mask_max_indices_rl = max(flip(mask, dims=(1,)), dim=1, keepdim=True)
    mask_max_indices_rl = mask.size(dim=1) - mask_max_indices_rl
    mask_max_indices_rl[mask_max_values_rl == 0] = 0

    H = mask.size(dim=0)
    W = mask.size(dim=1)
    
    base_grid : 'Tensor' = stack(
        meshgrid(
            arange(W, dtype=torch_int, device=img.device),
            arange(H, dtype=torch_int, device=img.device),
            indexing='ij'
        ),
        dim=-1
    ) # WxHx2

    x, _ = base_grid.permute(1, 0, 2).unbind(-1)
    
    mask[x < mask_max_indices_lr] = True
    mask[x > mask_max_indices_rl] = True

    return mask