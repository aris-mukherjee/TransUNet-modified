import numpy as np
from torch.utils.data import Dataset
import config.params as exp_config
import utils



class NCI_dataset(Dataset):
    def __init__(self, args, imtr, gttr, batch_size = exp_config.batch_size, train_or_eval = 'train'):
        self.args = args  
        self.imtr = imtr
        self.gttr = gttr
        self.batch_size = batch_size
        self.train_or_eval = train_or_eval
        self.sample_list = []
        i = 0
        for batch in iterate_minibatches(self.args, self.imtr, self.gttr, batch_size = exp_config.batch_size, train_or_eval = 'train'):
            self.sample_list.append({'image': batch[0], 'label': batch[1]})
            i = i+1

    def __len__(self):     #allows to call len(dataset)
        return self.imtr.shape[0]

    def __getitem__(self, idx):   #allows to index specific items of the dataset
        for batch in iterate_minibatches(self.args, self.imtr, self.gttr, batch_size = exp_config.batch_size, train_or_eval = 'train'):
            image, label = batch[0], batch[1]
        

        sample = {'image': image, 'label': label}
        #sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


def iterate_minibatches(args, 
                        images,
                        labels,
                        batch_size,
                        train_or_eval = 'train'):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue
        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        
        x = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        # ===========================    
        # data augmentation (contrast changes + random elastic deformations)
        # ===========================      
        if args.da_ratio > 0.0:

            # ===========================    
            # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
            # ===========================             
            # 90 degree rotation for cardiac images as the orientation is fixed for all other anatomies.
            do_rot90 = args.dataset in ['HVHD', 'CSF', 'UHE']
            x, y = utils.do_data_augmentation(images = x,
                                              labels = y,
                                              data_aug_ratio = args.da_ratio,
                                              sigma = exp_config.sigma,
                                              alpha = exp_config.alpha,
                                              trans_min = exp_config.trans_min,
                                              trans_max = exp_config.trans_max,
                                              rot_min = exp_config.rot_min,
                                              rot_max = exp_config.rot_max,
                                              scale_min = exp_config.scale_min,
                                              scale_max = exp_config.scale_max,
                                              gamma_min = exp_config.gamma_min,
                                              gamma_max = exp_config.gamma_max,
                                              brightness_min = exp_config.brightness_min,
                                              brightness_max = exp_config.brightness_max,
                                              noise_min = exp_config.noise_min,
                                              noise_max = exp_config.noise_max,
                                              rot90 = do_rot90)

        x = np.expand_dims(x, axis=-1)
        
        yield x, y