"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from json import load as json_load
from os.path import join, dirname, abspath

from imageio.v2 import imread
from numpy import asarray, stack as np_stack, tan as np_tan, float32 as np_float32, concatenate as np_concatenate
from torch import no_grad, reshape, stack, broadcast_to, randint, arange, meshgrid, split, rand, ones, zeros, float32 as torch_float32, from_numpy, uint8 as torch_uint8, tensor
from torch.linalg import norm
from torch.utils.data import Dataset
import torch.nn.functional as F
from rays import get_rays_for_each_pixel

from .utils import Rays

def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    blender2opencv = asarray([[1.,  0.,  0.,  0.],
                              [0., -1.,  0.,  0.],
                              [0.,  0., -1.,  0.],
                              [0.,  0.,  0.,  1.]], dtype=np_float32)
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = join(
            dirname(abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = join(root_fp, subject_id)
    with open(
        join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json_load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = join(data_dir, frame["file_path"] + ".png")
        rgba = imread(fname)
        camtoworlds.append(asarray(frame["transform_matrix"], dtype=np_float32) @ blender2opencv)
        images.append(rgba)

    images = np_stack(images, axis=0)
    camtoworlds = np_stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np_tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


class SubjectLoader(Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
        "chair_8_views",
        "drums_8_views",
        "ficus_8_views",
        "hotdog_8_views",
        "lego_8_views",
        "materials_8_views",
        "mic_8_views",
        "ship_8_views",
        "chair_8_views",
        "drums_4_views",
        "ficus_4_views",
        "hotdog_4_views",
        "lego_4_views",
        "materials_4_views",
        "mic_4_views",
        "ship_4_views",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np_concatenate([_images_train, _images_val])
            self.camtoworlds = np_concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        self.images = from_numpy(self.images).to(torch_uint8)
        self.camtoworlds = from_numpy(self.camtoworlds).to(torch_float32)
        self.K = tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch_float32,
        )  # (3, 3)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.images)

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

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = meshgrid(
                arange(self.WIDTH, device=self.images.device),
                arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x[None, :]
            y = y[None, :]

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        
        if self.training:
            x = reshape(x, (num_rays,1,1))
            y = reshape(y, (num_rays,1,1))
        
        origins, directions = get_rays_for_each_pixel(x.to(torch_float32), y.to(torch_float32), c2w, self.K[None])
        viewdirs = directions / norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            rays = Rays(origins=origins[:,0,0], viewdirs=viewdirs[:,0,0])
        else:
            rays = Rays(origins=origins[0], viewdirs=viewdirs[0])

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }