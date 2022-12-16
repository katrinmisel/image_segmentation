from keras.utils import Sequence
import numpy as np
import cv2
from helpers import *
import segmentation_models as sm
from keras.utils import load_img, img_to_array

class DataGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, backbone, dims, augmentation=False, shuffle=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.backbone = backbone
        self.dims = dims
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / int(self.batch_size)))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x_paths = [self.image_filenames[idx] for idx in indexes]
        batch_y_paths = [self.labels[idx] for idx in indexes]

        batch_x = [cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), (self.dims, self.dims)) for path_X in batch_x_paths]
        batch_y = [cv2.resize((cv2.imread(path_y,0)), (self.dims, self.dims)) for path_y in batch_y_paths]

        if self.augmentation:

            batch_x_aug, batch_y_aug = augment_batch(batch_x, batch_y)

            batch_x.extend(batch_x_aug)
            batch_y.extend(batch_y_aug)

        if self.backbone=='vgg16':
            preprocess_input = sm.get_preprocessing(self.backbone)
            batch_x = np.asarray(batch_x)
            preprocess_input(batch_x)

        elif self.backbone=='none': batch_x = batch_x

        else: 
            preprocess_input = sm.get_preprocessing(self.backbone)
            batch_x = preprocess_input(batch_x)

        return np.array(batch_x), np.array([coarsify_ohe(mask) for mask in batch_y])

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)