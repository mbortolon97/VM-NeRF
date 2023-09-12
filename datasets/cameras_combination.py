from torch import concat, meshgrid, arange, stack, int64, norm, ones
from torch.utils.data import Dataset
from geometrics import check_if_cameras_are_singular

class CamerasCombination(Dataset):
    def __init__(self, dataset, maximum_distance : 'float' = 5.0):
        self.dataset = dataset
        self.__maximum_distance = maximum_distance

        self.compute_the_valid_combinations()
    
    def compute_the_valid_combinations(self):
        first, second = meshgrid(arange(0, len(self.dataset), 1, dtype=int64), arange(0, len(self.dataset), 1, dtype=int64), indexing='ij')
        all_possible_combinations = stack((first, second), -1).view(-1, 2)

        valid_combinations = []
        for combination in all_possible_combinations:
            if combination[0] == combination[1]:
                continue
            
            camera_1_img_shape = self.dataset.images[combination[0]].shape
            camera_2_img_shape = self.dataset.images[combination[1]].shape
            
            camera_1_c2w = self.dataset.camtoworlds[combination[0]]
            camera_2_c2w = self.dataset.camtoworlds[combination[1]]

            camera_1_pos = camera_1_c2w[:3, 3]
            camera_2_pos = camera_2_c2w[:3, 3]

            if hasattr(self.dataset, "Ks"):
                camera_1_K = self.dataset.Ks[combination[0]]
                camera_2_K = self.dataset.Ks[combination[1]]
            else:
                camera_1_K = self.dataset.K
                camera_2_K = self.dataset.K

            if self.__maximum_distance < norm(camera_1_pos - camera_2_pos, p=2):
                continue

            if check_if_cameras_are_singular(camera_1_K, camera_1_c2w, camera_1_img_shape, camera_2_K, camera_2_c2w, camera_2_img_shape):
                continue
            
            valid_combinations.append(combination)
        
        self.valid_combinations_arr = stack(valid_combinations)
        self.valid_combinations_arr = self.valid_combinations_arr.contiguous()
    
    def __len__(self):
        return self.valid_combinations_arr.shape[0]
    
    def valid_combinations(self):
        return self.valid_combinations_arr.shape[0]

    def __getitem__(self, idx):
        extracted_combination = self.valid_combinations_arr[idx]
        return {
            'camera_1_idx': extracted_combination[0],
            'camera_2_idx': extracted_combination[1],
        }