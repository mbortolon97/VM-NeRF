from torch import tensor, Tensor, cos, sin, inverse, logical_and, divide, meshgrid, arange, cat, reshape, ones, zeros, eye, stack, sqrt, square, norm, sum, float32 as torch_float32, multiply, arctan, flip
from math import prod
from rays import get_rays_for_each_pixel

def trans_t(t : 'Tensor', device='cpu') -> 'Tensor':
    return tensor([
        [1.,           0.,            0.,             0.],
        [0.,           1.,            0.,             0.],
        [0.,           0.,            1.,              t],
        [0.,           0.,            0.,             1.]], dtype=t.dtype, device=device)

def rot_phi(phi : 'Tensor', device='cpu') -> 'Tensor':
    return tensor([
        [1.,           0.,            0.,             0.],
        [0.,           cos(phi),-sin(phi),0.],
        [0.,           sin(phi), cos(phi),0.],
        [0.,           0.,            0.,             1.]], dtype=phi.dtype, device=device)

def rot_theta(th : 'Tensor', device='cpu') -> 'Tensor':
    return tensor([
        [cos(th),0.,            -sin(th),0.],
        [0.,           1.,            0.,            0.],
        [sin(th),0.,            cos(th), 0.],
        [0.,           0.,            0.,            1.]], dtype=th.dtype, device=device)

def check_if_camera_are_visible_another_camera(camera_to_check_c2w, acquiring_camera_c2w, acquiring_camera_k, acquiring_camera_shape):
    result_point = acquiring_camera_k @ inverse(acquiring_camera_c2w)[:3] @ camera_to_check_c2w[:4, [3]]
    result_point = divide(result_point, result_point[2]).ravel()[:2]

    return logical_and(result_point < tensor(acquiring_camera_shape)[:2], result_point > 0).all()

def check_if_cameras_are_singular(camera_1_k, camera_1_c2w, camera_1_shape, camera_2_k, camera_2_c2w, camera_2_shape):
    return check_if_camera_are_visible_another_camera(camera_1_c2w, camera_2_c2w, camera_2_k, camera_2_shape) and check_if_camera_are_visible_another_camera(camera_2_c2w, camera_1_c2w, camera_1_k, camera_1_shape)

def sphere_depth_to_3D_points(depth, K_matrix, bb=[]):
    if len(bb) != 4:
        bb = tensor([0., 0., depth.size(dim=1), depth.size(dim=0)], dtype=depth.dtype)
    
    minx = bb[0]
    miny = bb[1]
    maxx = bb[2]
    maxy = bb[3]
    
    base_grid : 'Tensor' = stack(
        meshgrid(
            arange(minx, maxx, dtype=depth.dtype, device=depth.device),
            arange(miny, maxy, dtype=depth.dtype, device=depth.device),
            indexing='ij'
        ),
        dim=-1
    )  # WxHx2
    
    i, j = (base_grid.permute(1, 0, 2).unsqueeze(0) + 0.5).unbind(-1)

    rays_o, rays_d = get_rays_for_each_pixel(i, j, eye(4, dtype=depth.dtype, device=depth.device)[None, ...], K_matrix[None, ...])

    points_3D = rays_o + multiply(rays_d, depth[None])

    return points_3D.view(-1, 3)

def image_plane_depth_to_3D_points(depth, K_matrix, bb=[]):
    if len(bb) != 4:
        bb = tensor([0., 0., depth.size(dim=1), depth.size(dim=0)], device=depth.device)
    
    minx = bb[0]
    miny = bb[1]
    maxx = bb[2]
    maxy = bb[3]

    base_grid : 'Tensor' = stack(
        meshgrid(
            arange(minx, maxx, dtype=torch_float32, device=depth.device),
            arange(miny, maxy, dtype=torch_float32, device=depth.device),
            indexing='ij'
        ),
        dim=-1
    )  # WxHx2

    i, j = base_grid.permute(1, 0, 2).unbind(-1)
    
    K_matrix_inverse = inverse(cat((cat((K_matrix, zeros((K_matrix.shape[0], 1), dtype=K_matrix.dtype, device=depth.device)), dim=-1), eye(4)[[-1], :]), dim=0))
    points_2D = cat((reshape(i, (-1, 1)).type(K_matrix.dtype), reshape(j, (-1, 1)).type(K_matrix.dtype), ones((prod(depth.size()), 1), dtype=K_matrix.dtype, device=depth.device), divide(1., depth.view(-1, 1))), -1)
    
    point_3D = (K_matrix_inverse @ points_2D.T).T

    return divide(point_3D, point_3D[:, [-1]] + 1e-15)[:, 0:3]

def image_plane_to_sphere_depth(depth, K_matrix, bb=[]):
    points_3D = image_plane_depth_to_3D_points(depth, K_matrix, bb=bb)
    
    if len(bb) != 4:
        bb = tensor([0., 0., depth.size(dim=1), depth.size(dim=0)], dtype=depth.dtype)
    
    minx = bb[0]
    miny = bb[1]
    maxx = bb[2]
    maxy = bb[3]

    resulting_depth = points_3D_to_sphere_depth(points_3D, K_matrix, minx, maxx, miny, maxy)

    return resulting_depth.view(*depth.shape)

def sphere_to_image_plane_depth(input_depth_warp, K_matrix, bb=[]):
    if len(bb) != 4:
        bb = tensor([0., 0., input_depth_warp.size(dim=1), input_depth_warp.size(dim=0)], dtype=input_depth_warp.dtype, device=input_depth_warp.device)
    
    minx = bb[0]
    miny = bb[1]
    maxx = bb[2]
    maxy = bb[3]

    x,y = meshgrid(arange(minx, maxx, 1, dtype=torch_float32, device=input_depth_warp.device), arange(miny, maxy, 1, dtype=torch_float32, device=input_depth_warp.device), indexing='xy')
    
    points = cat((reshape(x, (-1, 1)), reshape(y, (-1, 1)), ones((prod(x.shape), 1), dtype=input_depth_warp.dtype, device=input_depth_warp.device)), -1)
    metric_points = (inverse(K_matrix) @ points.T).T

    z_prime = multiply(reshape(input_depth_warp, (-1, 1)), cos(arctan(divide(sqrt(sum(square(metric_points[:,0:2]), -1, True)), metric_points[:,[2]]))))
    z_prime = reshape(z_prime, input_depth_warp.size())
    
    return z_prime

def points_3D_to_sphere_depth(points_3D : 'Tensor', K_matrix : 'Tensor', minx : 'int', maxx : 'int', miny : 'int', maxy : 'int') -> 'Tensor':
    base_grid : 'Tensor' = stack(
        meshgrid(
            arange(minx, maxx, dtype=points_3D.dtype, device=points_3D.device),
            arange(miny, maxy, dtype=points_3D.dtype, device=points_3D.device),
            indexing='ij'
        ),
        dim=-1
    )  # WxHx2
    
    i, j = base_grid.permute(1, 0, 2).unbind(-1)

    _, rays_d = get_rays_for_each_pixel(i[None, ...], j[None, ...], eye(4, dtype=points_3D.dtype, device=points_3D.device)[None, ...], K_matrix[None, ...])
    depth_scaling = norm(rays_d, dim=-1)
    
    resulting_depth_sqrt = sqrt(sum(square(points_3D), dim=-1, keepdim=True))

    resulting_depth_sqrt = divide(resulting_depth_sqrt, depth_scaling.view(-1, 1))

    return resulting_depth_sqrt

def sphere_depth_to_image_plane_depth(depth, K_matrix, bb=[]):
    pts_3D = sphere_depth_to_3D_points(depth, K_matrix, bb)

    rotation_matrix = eye(3, dtype=pts_3D.dtype, device=pts_3D.device)
    rotation_matrix[1,1] = -rotation_matrix[1,1]
    rotation_matrix[2,2] = -rotation_matrix[2,2]

    pts_3D = (rotation_matrix @ pts_3D.T).T
    
    z_prime = pts_3D[:, -1].view(depth.shape) + 1e-10
    
    if False:
        from torch import mean
        import open3d as o3d
        sphere_points = sphere_depth_to_3D_points(depth, K_matrix, bb)  

        plane_points = image_plane_depth_to_3D_points(z_prime, K_matrix, bb)

        plane_points = (rotation_matrix @ plane_points.T).T

        per_point_error = sum(square(sphere_points - plane_points), -1) / sphere_points.shape[-1]
        mean_error = mean(per_point_error)
        
        print("Mean error translation: ", mean_error)

        # show the results
        pcd_sphere = o3d.geometry.PointCloud()
        pcd_sphere.points = o3d.utility.Vector3dVector(sphere_points.cpu().numpy())
        pcd_sphere.paint_uniform_color([1., 0., 0.])

        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(plane_points.cpu().numpy())
        pcd_plane.paint_uniform_color([0., 0., 1.])

        o3d.visualization.draw_geometries([pcd_sphere, pcd_plane])
    
    return z_prime