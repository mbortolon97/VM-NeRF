from torch import Tensor, exp, ones_like, zeros_like, clamp, rand_like, tensor, ones, bool as torch_bool, float32 as torch_float32
from nerfacc import Grid
from typing import Callable, Tuple, Union, Optional
from nerfacc.intersection import ray_aabb_intersect
from nerfacc.contraction import ContractionType
import nerfacc.cuda as _C
from nerfacc.vol_rendering import render_visibility

def ray_marching(
    rays_o : Tensor,
    rays_d : Tensor,
    t_min: Optional[Tensor] = None,
    t_max: Optional[Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[Tensor] = None,
    grid: Optional[Grid] = None,
    # sigma/alpha function for skipping invisible space
    sigma_fn: Optional[Callable] = None,
    alpha_fn: Optional[Callable] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> 'Tuple[Tensor, Tensor, Tensor]':
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if alpha_fn is not None and sigma_fn is not None:
        raise ValueError(
            "Only one of `alpha_fn` and `sigma_fn` should be provided."
        )

    # logic for t_min and t_max:
    # 1. if t_min and t_max are given, use them with highest priority.
    # 2. if t_min and t_max are not given, but scene_aabb is given, use
    # ray_aabb_intersect to compute t_min and t_max.
    # 3. if t_min and t_max are not given, and scene_aabb is not given,
    # set t_min to 0.0, and t_max to 1e10. (the case of unbounded scene)
    # 4. always clip t_min with near_plane and t_max with far_plane if given.
    if t_min is None or t_max is None:
        if scene_aabb is not None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, scene_aabb)
        else:
            t_min = zeros_like(rays_o[..., 0])
            t_max = ones_like(rays_o[..., 0]) * 1e10
    if near_plane is not None:
        t_min = clamp(t_min, min=near_plane)
    if far_plane is not None:
        t_max = clamp(t_max, max=far_plane)

    # stratified sampling: prevent overfitting during training
    if stratified:
        t_min = t_min + rand_like(t_min) * render_step_size
    
    # use grid for skipping if given
    if grid is not None:
        grid_roi_aabb = grid.roi_aabb
        grid_binary = grid.binary
        contraction_type = grid.contraction_type.to_cpp_version()
    else:
        grid_roi_aabb = tensor(
            [-1e10, -1e10, -1e10, 1e10, 1e10, 1e10],
            dtype=torch_float32,
            device=rays_o.device,
        )
        grid_binary = ones(
            [1, 1, 1], dtype=torch_bool, device=rays_o.device
        )
        contraction_type = ContractionType.AABB.to_cpp_version()
    
    # marching with grid-based skipping
    packed_info, ray_indices, t_starts, t_ends, num_samples = _C.ray_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # coontraction and grid
        grid_roi_aabb.contiguous(),
        grid_binary.contiguous(),
        contraction_type,
        # sampling
        render_step_size,
        cone_angle,
    )
    
    # skip invisible space
    if sigma_fn is not None or alpha_fn is not None:
        # Query sigma without gradients
        if sigma_fn is not None:
            sigmas = sigma_fn(t_starts, t_ends, ray_indices)
            assert (
                sigmas.shape == t_starts.shape
            ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
            alphas = 1.0 - exp(-sigmas * (t_ends - t_starts))
        elif alpha_fn is not None:
            alphas = alpha_fn(t_starts, t_ends, ray_indices)
            assert (
                alphas.shape == t_starts.shape
            ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)

        # Compute visibility of the samples, and filter out invisible samples
        masks = render_visibility(
            alphas,
            ray_indices=ray_indices,
            packed_info=packed_info,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            n_rays=rays_o.shape[0],
        )
        ray_indices, t_starts, t_ends = (
            ray_indices[masks],
            t_starts[masks],
            t_ends[masks],
        )

    return ray_indices, t_starts, t_ends