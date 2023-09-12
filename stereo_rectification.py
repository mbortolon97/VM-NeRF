from numpy import asarray
from sys import float_info

from torch import inverse, sign, Tensor, det, norm, divide, roll, diag, flatten, logical_or, floor, ceil, reshape, multiply, cat, stack, matrix_rank, logical_not, ones, tensor, min as torch_min, max as torch_max, arange, meshgrid, int as torch_int, amin, amax, hstack, cross, clone, abs as torch_abs, dot, zeros

from torch.linalg import qr
from typing import Tuple

def art(P):
    """
    ART  Factorize camera matrix into intrinsic and extrinsic matrices

    [A,R,t] = art(P,fsign)  factorize the projection matrix P
    as P=A*[R;t] and enforce the sign of the focal lenght to be fsign.
    By default fsign=1.

    Author: A. Fusiello, 1999

    The function expect positive focal length
    """

    s = P[0:3,[3]]
    Q = inverse(P[0:3, 0:3])
    U,B = qr(Q)

    sig = sign(B[2,2])
    B = B * sig
    s = s * sig

    if B[0,0] < 0:
        E = tensor([
            [-1., 0., 0.],
            [0.,-1.,0.],
            [0.,0.,1.]
        ], dtype=P.dtype, device=P.device)
        B = E @ B
        U = U @ E
    
    if B[1,1] < 0:
        E = tensor([
            [1., 0., 0.],
            [0.,-1.,0.],
            [0.,0.,1.]
        ], dtype=P.dtype, device=P.device)
        B = E @ B
        U = U @ E
    
    if det(U) < 0:
        U = -U
        s = -s
    
    if norm(Q-U @ B) > 1e-10 and norm(Q+U @ B) > 1e-10:
        raise ValueError('Something wrong with the QR factorization')
    
    R = U.T
    t = B @ s
    A = inverse(B)
    A = divide(A, A[2,2])

    # sanity check
    if det(R) < 0.:
        raise ValueError('R is not a rotation matrix')
    if A[2,2] < 0.:
        raise ValueError('Wrong sign of A[2,2]')
    W = A @ cat((R, t), axis=1)
    if matrix_rank(stack((flatten(P), flatten(W)), axis=-1)) != 1:
        raise ValueError('something is wrong with the ART factorization')
    
    return A, R, t

def skew(v):
    if len(v) == 4: v = v[:3]/v[3]
    skv = roll(roll(diag(flatten(v)), 1, 1), -1, 0)
    return skv - skv.T

def linear_interpolation_2D_grid(original_img, query_points):
    outside_boundaries = logical_or(logical_or(query_points[:,0] >= original_img.shape[1] - 1., query_points[:,0] <= 1.), logical_or(query_points[:,1] >= original_img.shape[0] - 1., query_points[:,1] <= 1.))

    query_points[outside_boundaries,:] = 1.5
    xmin = floor(query_points[:,0])
    xmax = ceil(query_points[:,0] + 1e-10)
    ymin = floor(query_points[:,1])
    ymax = ceil(query_points[:,1] + 1e-10)

    x_limits_diff = xmax - xmin + 1e-10
    y_limits_diff = ymax - ymin + 1e-10

    xmin = xmin.long()
    xmax = xmax.long()
    ymin = ymin.long()
    ymax = ymax.long()

    y1 = multiply(reshape(divide(xmax - query_points[:,0] + 1e-10, x_limits_diff), (-1,1)), original_img[ymin, xmin]) + multiply(reshape(divide(query_points[:,0] - xmin, x_limits_diff), (-1,1)), original_img[ymin, xmax])
    y2 = multiply(reshape(divide(xmax - query_points[:,0] + 1e-10, x_limits_diff), (-1,1)), original_img[ymax, xmin]) + multiply(reshape(divide(query_points[:,0] - xmin + 1e-10, x_limits_diff), (-1,1)), original_img[ymax, xmax])
    valid_points = logical_not(outside_boundaries)
    interpolation = multiply(multiply(reshape(divide(ymax - query_points[:,1] + 1e-10, y_limits_diff), (-1,1)), y1) + multiply(reshape(divide(query_points[:,1] - ymin + 1e-10, y_limits_diff), (-1,1)), y2), reshape(valid_points.float(), (-1,1)))

    return interpolation, valid_points

def p2t(H,m):
    if H.size(dim=0) == 3 and H.size(dim=1) == 3 and len(H.size()) == 0:
        raise ValueError("Invalid input transformation")
    
    if H.size(dim=1) == 2:
        raise ValueError("Image coordinate must be cartesian")

    points_3d = cat((m, ones((m.size(dim=0), 1), dtype=m.dtype, device=m.device)), 1)
    transformed_points = (H @ points_3d.T).T
    return divide(transformed_points[:,0:2], transformed_points[:,[2]])[:,0:2]

def imwarp(img : 'Tensor', H : 'Tensor', sz : 'str' ='same', not_valid_value : 'float' =8.):
    """
    Image Warping
    I2 = imwarp(I,H) apply the projective transformation specified by H to
    the image I using linear interpolation. The output image I2 has the
    same size of I.
    
    I2 = imwarp(I,H,meth) use method 'meth' for interpolation (see interp2
    for the list of options).
    
    I2 = imwarp(I,H,meth,sz) yield an output image with specific size. sz
    can be:
    
        - 'valid' Make output image I2 large enough to contain the entire rotated image.
        - 'same'  Make output image I2 the same size as the input image I, cropping the warped image to fit (default).
        -  a vector of 4 elements specifying the bounding box
    
    The output bb is the bounding box of the transformed image in the
    coordinate frame of the input image. The first 2 elements of the bb are
    the translation that have been applied to the upper left corner.
    
    The bounding box is specified with [minx; miny; maxx; maxy];
    
    See also: INTERP2
    """
    if H.size(dim=0) != 3 or H.size(dim=1) != 3:
        raise ValueError("Invalid input transformation")
    
    if isinstance(sz, str):
        if sz == 'same':
            # same bb as the input image
            minx = 0
            maxx = img.size(dim=1)-1
            miny = 0
            maxy = img.size(dim=0)-1
        elif sz == 'valid':
            corners = tensor([
                [0., 0.],
                [0., img.size(dim=0)],
                [img.size(dim=1), 0.],
                [img.size(dim=1), img.size(dim=0)]
            ], dtype=img.dtype, device=img.device)
            corners_x=p2t(H,corners)

            minx = floor(torch_min(corners_x[:,0]))
            maxx = ceil(torch_max(corners_x[:,0]))
            miny = floor(torch_min(corners_x[:,1]))
            maxy = ceil(torch_max(corners_x[:,1]))
    elif len(sz) == 4:
        minx = sz[0]
        miny = sz[1]
        maxx = sz[2]
        maxy = sz[3]
    else:
        raise ValueError('invalid size option')
    
    bb = tensor([minx, miny, maxx, maxy], dtype=img.dtype, device=img.device)
    x,y = meshgrid(arange(minx, maxx, 1, dtype=torch_int, device=img.device), arange(miny, maxy, 1, dtype=torch_int, device=img.device), indexing='xy')
    
    original_points = stack((flatten(x), flatten(y)), axis=-1).type(H.dtype)
    query_points = p2t(inverse(H), original_points)

    interpolation_results, valid_points = linear_interpolation_2D_grid(img, query_points)
    interpolation_results[logical_not(valid_points)] = not_valid_value
    I2 = reshape(interpolation_results, ((maxy-miny).int(), (maxx-minx).int(), interpolation_results.size(dim=-1)))
    # I2 = I2.transpose(1, 0, 2)
    I2 = I2.type(img.dtype)
    valid_points = reshape(valid_points, ((maxy-miny).int(), (maxx-minx).int()))

    return I2, valid_points * 1., bb, reshape(query_points, ((maxy-miny).int(), (maxx-minx).int(), 2))

def fund(pml, pmr):
    """
    FUND Computes fundamental matrix and epipoles from camera matrices.
    
    [F,el,er] = fund(pml,pmr) calcola la matrice fondamentale
    F, l'epipolo sinistro el e destro er, partendo dalle due 
    matrici  di proiezione prospettica pml (MPP sinistra) e 
    pmr (MPP destra).
    """

    # calcolo i centri ottici dalle due MPP
    cl = -inverse(pml[:,0:3]) @ pml[:,[3]]
    cr = -inverse(pmr[:,0:3]) @ pmr[:,[3]]

    # calcolo gli epipoli come proiezione dei centri ottici
    el = pml @ cat((cr.T, ones((cr.shape[1], 1), dtype=pml.dtype, device=pml.device)), 1).T
    er = pmr @ cat((cl.T, ones((cl.shape[1], 1), dtype=pmr.dtype, device=pmr.device)), 1).T

    # computation of the fundamental matrix
    F = skew(er) @ pmr[:,0:3] @ inverse(pml[:,0:3])

    F = divide(F, norm(F))

    return F, el, er

def get_rectify_axis(c1 : 'Tensor', c2 : 'Tensor', w2c_1 : 'Tensor') -> 'Tuple[Tensor, Tensor, Tensor]':
    v1 = (c2 - c1).flatten()
    
    k = tensor([[1.],[1.],[0.]], dtype=c1.dtype, device=c1.device)
    u = tensor([[-1.],[1.],[0.]], dtype=c1.dtype, device=c1.device)

    k = inverse(w2c_1[:, 0:3]) @ k
    u = inverse(w2c_1[:, 0:3]) @ u

    k_result = dot(k.flatten(), v1)
    k_result_sign = divide(k_result, torch_abs(k_result))
    u_result = dot(u.flatten(), v1)
    u_result_sign = divide(u_result, torch_abs(u_result))

    sign_sum = k_result_sign + u_result_sign

    alpha_y_inversion = torch_abs(sign_sum)
    alpha_y_inversion = divide(alpha_y_inversion, alpha_y_inversion + 1e-10)

    # new y axes (orthogonal to old z and new x)
    v2 = cross(flatten(w2c_1[[2],0:3]), v1)
    
    new_x_axis = multiply(1. - alpha_y_inversion, v1) + multiply(alpha_y_inversion, v2)
    new_y_axis = multiply(1. - alpha_y_inversion, v2) + multiply(alpha_y_inversion, v1)

    # new z axis (no choice, orthogonal to baseline and y)
    new_z_axis = cross(v1, v2)

    ori_x_axis = tensor([[1.],[0.],[0.]], dtype=c1.dtype, device=c1.device)
    ori_x_axis = inverse(w2c_1[:, 0:3]) @ ori_x_axis
    
    dot_result = dot(ori_x_axis.flatten(), new_x_axis)

    new_x_axis = multiply(divide(dot_result, torch_abs(dot_result)), new_x_axis)

    ori_y_axis = tensor([[0.],[1.],[0.]], dtype=c1.dtype, device=c1.device)
    ori_y_axis = inverse(w2c_1[:, 0:3]) @ ori_y_axis

    dot_result = dot(ori_y_axis.flatten(), new_y_axis)

    new_y_axis = multiply(divide(dot_result, torch_abs(dot_result)), new_y_axis)
    
    return new_x_axis, new_y_axis, new_z_axis

def rectify_from_components(A1,R1,t1,A2,R2,t2,d1 = asarray([0., 0.]),d2 = asarray([0., 0.])):
    w2c_1 = cat((R1, t1), 1)
    w2c_2 = cat((R2, t2), 1)

    # projection matrix
    Po1 = A1 @ w2c_1
    Po2 = A2 @ w2c_2
    
    # optical centers (unchanged)
    c1 = - R1.T @ inverse(A1) @ Po1[:,[3]]
    c2 = - R2.T @ inverse(A2) @ Po2[:,[3]]

    # new x axis (baseline from c1 to c2)
    v1, v2, v3 = get_rectify_axis(c1, c2, w2c_1)

    # new extrinsic (translation unchanged)
    R = stack((divide(v1, norm(v1)), divide(v2, norm(v2)), divide(v3, norm(v3))))

    # new intrinsic (arbitrary)
    An1 = clone(A2)
    # An1[0,1]=0.
    An2 = clone(A2)
    # An2[0,1]=0.

    An1[0,2] = An1[0,2] + d1[0]
    An1[1,2] = An1[1,2] + d1[1]
    An2[0,2] = An2[0,2] + d2[0]
    An2[1,2] = An2[1,2] + d2[1]

    P_t1 = -R @ c1
    P_t2 = -R @ c2
    Pn1 = An1 @ cat((R, t1), 1)
    Pn2 = An2 @ cat((R, t2), 1)

    T1 = Pn1[0:3, 0:3] @ inverse(Po1[0:3, 0:3])
    T2 = Pn2[0:3, 0:3] @ inverse(Po2[0:3, 0:3])

    return T1, T2, Pn1, Pn2, An1, An2, R, P_t1, P_t2

def rectify(Po1,Po2,d1 = asarray([0., 0.]),d2 = asarray([0., 0.])):
    if not ((d1[1] - d2[1] < float_info.epsilon and d1[0] - d2[0] > float_info.epsilon) or (d1[1] - d2[1] > float_info.epsilon and d1[0] - d2[0] < float_info.epsilon) or (d1[1] - d2[1] < float_info.epsilon and d1[0] - d2[0] < float_info.epsilon)):
        raise ValueError('left and right vertical displacements must be the same')

    # factorise old PPM
    A1, R1, t1 = art(Po1)
    A2, R2, t2 = art(Po2)
    return rectify_from_components(A1,R1,t1,A2,R2,t2,d1=d1,d2=d2)

def minimum_common_bounding_box(s1, s2, H1, H2):
    corners = tensor([
        [0., 0.],
        [0., s1[0]],
        [s1[1], 0.],
        [s1[1], s1[0]]
    ], dtype=H1.dtype, device=H1.device)
    
    corners_x = p2t(H1, corners)

    minx = floor(torch_min(corners_x[:,0]))
    maxx = ceil(torch_max(corners_x[:,0]))
    miny = floor(torch_min(corners_x[:,1]))
    maxy = ceil(torch_max(corners_x[:,1]))

    bb1 = tensor([[minx],[miny],[maxx],[maxy]], dtype=H1.dtype, device=H1.device)

    corners = tensor([
        [0., 0.],
        [0., s2[0]],
        [s2[1], 0.],
        [s2[1], s2[0]],
    ], dtype=H1.dtype, device=H1.device)

    corners_x = p2t(H2, corners)

    minx = floor(torch_min(corners_x[:,0]))
    maxx = ceil(torch_max(corners_x[:,0]))
    miny = floor(torch_min(corners_x[:,1]))
    maxy = ceil(torch_max(corners_x[:,1]))

    bb2 = tensor([[minx],[miny],[maxx],[maxy]], dtype=H1.dtype, device=H1.device)

    q1 = amin(cat((bb1, bb2), axis=1), 1)
    q2 = amax(cat((bb1, bb2), axis=1), 1)

    return hstack((q1[0:2], q2[2:4]))

def compute_stereo_rectification_from_components(K1, R1, t1, K2, R2, t2, IL_shape, IR_shape):
    pml = K1 @ cat((R1, t1), -1)
    pmr = K2 @ cat((R2, t2), -1)

    F, epipole_left, epipole_right = fund(pml, pmr)

    TL, TR, pml1, pmr1, _, _, _, _, _ = rectify_from_components(K1, R1, t1, K2, R2, t2)

    # centering LEFT image
    p = tensor([[IL_shape[0]/2],[IL_shape[1]/2],[1.]], dtype=K1.dtype, device=K1.device)
    px = TL @ p
    dL = p[0:2] - divide(px[0:2], px[2])

    # centering RIGHT image
    p = tensor([[IR_shape[0]/2],[IR_shape[1]/2],[1.]], dtype=K2.dtype, device=K2.device)
    px = TR @ p
    dR = p[0:2] - divide(px[0:2], px[2])

    dL[1] = dR[1]

    TL1, TR1, pml1, pmr1, An1, An2, R, t1, t2 = rectify_from_components(K1, R1, t1, K2, R2, t2, dL, dR)

    return F, TL1, TR1, pml1, pmr1, epipole_left, epipole_right, An1, An2, R, t1, t2, dL, dR

def compute_stereo_rectification(pml, pmr, IL_shape, IR_shape):
    F, epipole_left, epipole_right = fund(pml, pmr)

    TL, TR, pml1, pmr1, _, _, _, _, _ = rectify(pml, pmr)

    # centering LEFT image
    p = tensor([[IL_shape[0]/2],[IL_shape[1]/2],[1.]], dtype=pml.dtype, device=pml.device)
    px = TL @ p
    dL = p[0:2] - divide(px[0:2], px[2])

    # centering RIGHT image
    p = tensor([[IR_shape[0]/2],[IR_shape[1]/2],[1.]], dtype=pml.dtype, device=pml.device)
    px = TR @ p
    dR = p[0:2] - divide(px[0:2], px[2])

    dL[1] = dR[1]
    
    TL1, TR1, pml1, pmr1, An1, An2, R, t1, t2 = rectify(pml, pmr, dL, dR)

    return F, TL1, TR1, pml1, pmr1, epipole_left, epipole_right, An1, An2, R, t1, t2, dL, dR

def warp_image(IL, IR, TL, TR):
    bb = minimum_common_bounding_box(IL.size(), IR.size(), TL, TR)

    JL, alphaL, bbL, _ = imwarp(IL, TL, sz=bb, not_valid_value=0.)
    JR, alphaR, bbR, _ = imwarp(IR, TR, sz=bb, not_valid_value=0.)

    return JL, JR, bb, bbL, bbR, alphaL, alphaR

def cut_to_original_image(image, dest_K : 'Tensor', input_K : 'Tensor', target_K : 'Tensor', input_img_shape : 'Tuple[int, int]', target_img_shape : 'Tuple[int, int]', s : 'float', bb : 'Tensor'):
    minx = bb[0].item()
    miny = bb[1].item()
    
    input_x_before = input_K[0, 2].item()
    input_x_after = (input_img_shape[1] - input_K[0, 2]).item()
    input_y_before = input_K[1, 2].item()
    input_y_after = (input_img_shape[0] - input_K[1, 2]).item()

    target_x_before = target_K[0, 2].item()
    target_x_after = (target_img_shape[1] - target_K[0, 2]).item()
    target_y_before = target_K[1, 2].item()
    target_y_after = (target_img_shape[0] - target_K[1, 2]).item()

    x_before = s * (target_x_before - input_x_before) + input_x_before
    y_before = s * (target_y_before - input_y_before) + input_y_before
    x_after = s * (target_x_after - input_x_after) + input_x_after
    y_after = s * (target_y_after - input_y_after) + input_y_after

    dest_pp_x = dest_K[0, 2].item()
    dest_pp_y = dest_K[1, 2].item()

    bb_ll_x = dest_pp_x - x_before - minx
    bb_ll_y = dest_pp_y - y_before - miny

    bb_ll_x_correct = max(0.0, bb_ll_x)
    bb_ll_y_correct = max(0.0, bb_ll_y)
    
    bb_ul_x = dest_pp_x + x_after - minx
    bb_ul_y = dest_pp_y + y_after - miny

    bb_ul_x_correct = min(image.shape[1], bb_ul_x)
    bb_ul_y_correct = min(image.shape[0], bb_ul_y)

    out_image = image[round(bb_ll_y_correct):round(bb_ul_y_correct), round(bb_ll_x_correct):round(bb_ul_x_correct)]

    out_image = cat((zeros((out_image.shape[0], round(abs(bb_ll_x - bb_ll_x_correct)), out_image.shape[2]), dtype=out_image.dtype, device=out_image.device), out_image, zeros((out_image.shape[0], round(abs(bb_ul_x - bb_ul_x_correct)), out_image.shape[2]), dtype=out_image.dtype, device=out_image.device)), dim=1)
    out_image = cat((zeros((round(abs(bb_ll_y - bb_ll_y_correct)), out_image.shape[1], out_image.shape[2]), dtype=out_image.dtype, device=out_image.device), out_image, zeros((round(abs(bb_ul_y - bb_ul_y_correct)), out_image.shape[1], out_image.shape[2]), dtype=out_image.dtype, device=out_image.device)), dim=0)
    
    return out_image