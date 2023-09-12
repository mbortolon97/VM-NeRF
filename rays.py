from typing import Tuple
from torch import Tensor, arange, meshgrid, stack, bmm, transpose, float32 as torch_float32, ones_like, divide, inverse, broadcast_to, rand

__all__ = [
    'get_rays'
]

def get_rays(intrinsicparams: 'Tensor', c2w: 'Tensor', K : 'Tensor', device : 'str') -> 'Tuple[Tensor,Tensor]':
    """
        This function given the parameter of a pinhole camera (width, height and focal along the x and y-axes) output the rays direction and origin

        intrinsicparams: is a 3 vector containing in pos 0 the height, in 1 the width, 2 the focal along the x axis and 3 along the y axis
        c2w is a 4x4 matrix containing the translation and rotation from the world to the camera
        expected_img (optional): this image contain the optional data format
    """

    H = int(intrinsicparams[0].item())
    W = int(intrinsicparams[1].item())
    
    increment = 0.5
    if False:
        increment = rand(grid_output.shape, dtype=grid_output.dtype, device=grid_output.device)

    base_grid : 'Tensor' = stack(
        meshgrid(
            arange(W, dtype=torch_float32, device=device) + increment,
            arange(H, dtype=torch_float32, device=device) + increment,
            indexing='ij'
        ),
        dim=-1
    ) # WxHx2 d
    
    grid_output = base_grid.permute(1, 0, 2).unsqueeze(0)

    i, j = grid_output.unbind(-1)

    return get_rays_for_each_pixel(i, j, c2w, K)

def get_rays_for_each_pixel(x : 'Tensor', y : 'Tensor', c2ws: 'Tensor', Ks : 'Tensor') -> 'Tuple[Tensor,Tensor]':
    coords = stack((x, y, ones_like(x)), -1) # (B, H, W, 3)
    cam_dirs = bmm(broadcast_to(inverse(Ks), (coords.shape[0], Ks.shape[2], Ks.shape[1])), coords.permute(0, 3, 1, 2).view(x.shape[0], 3, -1)).view(x.shape[0], 3, x.shape[1], x.shape[2]).permute(0, 2, 3, 1)
    # dirs = bmm(coords.view(x.shape[0], -1, 3), broadcast_to(inverse(K), (coords.shape[0], K.shape[2], K.shape[1]))).view(x.shape[0], 3, x.shape[1], x.shape[2]).permute(0, 2, 3, 1)
    # cam_dirs = stack([coord @ inverse(c).T for (coord, c) in zip(coords, Ks)])
    # dirs = divide(dirs, abs(dirs[...,[-1]]))
    # dirs = stack([(x-K[..., 0, 2])/K[..., 0, 0], (y-K[..., 1, 2])/K[..., 1, 1], ones_like(x)], -1) # (B, H, W, 3)
    dirs_ori_shape = cam_dirs.shape
    
    # rays_d = stack([v @ c2w[:3, :3].T for (v, c2w) in zip(cam_dirs, c2ws)])
    rays_d = bmm(cam_dirs.view(cam_dirs.shape[0], -1, 3), transpose(c2ws[..., :3, :3], -1, -2)).view(dirs_ori_shape)
    # rays_d = rays_d / norm(rays_d, dim=-1, keepdim=True)
    # rays_o = transpose(c2w[..., :3, [3]], -1, -2).expand(rays_d.shape)
    rays_o = broadcast_to(c2ws[:, None, None, :3, 3], rays_d.shape)
    
    return rays_o, rays_d

def ndc_rays(intrinsicparams: 'Tensor', near : 'float', rays_o : 'Tensor', rays_d : 'Tensor'):
    H = intrinsicparams[0]
    W = intrinsicparams[1]
    fx = intrinsicparams[2]
    fy = intrinsicparams[3]

    # Shift the origin to the near plane
    t = divide(near - rays_o[..., 2], rays_d[..., 2])
    rays_o = rays_o + t[..., None] * rays_d
    
    # Projection
    o0 = (fx / (W / 2.)) * (rays_o[..., 0] / rays_o[..., 2])
    o1 = (fy / -(H / 2.)) * (rays_o[..., 1] / rays_o[..., 2])
    o2 = 1. - ((2. * near) / rays_o[..., 2])
    # o0 = -1./(W/(2.*fx)) * rays_o[:, 0] / rays_o[:, 2]
    # o1 = -1./(H/(2.*fy)) * rays_o[:, 1]/ rays_o[:, 2]
    # o2 = 1. / 2. * near / rays_o[:, 2]

    d0 = (fx / (W / 2.)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = (fy / -(H / 2.)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = (2. * near) / rays_o[..., 2]

    rays_o = stack((o0, o1, o2), -1)
    rays_d = stack((d0, d1, d2), -1)

    return rays_o, rays_d

