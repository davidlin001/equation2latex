# Title: sampler.py
# Author: Cody Kala
# Date: 6/2/2018
# ====================
# Module that implements a Sampler for the Im2Latex-100k dataset.

import random
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class Im2LatexSampler(Sampler):
    """ Implements a PyTorch Sampler object to iterate over the images in
    the Im2Latex-100k dataset.

    This Sampler class is intended to be used on the preprocessed images
    generated by the preprocessing scripts of Deng et al. Their scripts
    can be found here:
    
      https://github.com/harvardnlp/im2markup

    In their preprocessing scripts, they

        1) crop images to remove extraneous whitespace
        2) pad images on all spatial dimensions
        3) group images with similar dimensions into buckets by padding
            with additional whitespace.

    For reference, Deng et al used the following dimensions to bucket the
    images. Dimensions are expressed as (width, height) tuples.

        (120, 50)
        (160, 40)
        (200, 40), (200, 50)
        (240, 40), (240, 50)
        (280, 40), (280, 50)
        (320, 40), (320, 50)
        (360, 40), (360, 50), (360, 60), (360, 100)
        (400, 50), (400, 60)
        (500, 100)

    To use different buckets for this Sampler class, the images should
    be preprocessed to the desired bucket sizes before initializing
    the Sampler.
    """

    def __init__(self, dataset, batch_size):
        """ Initializes the Sampler object.

        Inputs:
            dataset : torch.utils.data.Dataset object
                A Pytorch Dataset containing the examples for the dataset.
            batch_size : int
                The batch size.

        Outputs:
            None, but initializes the Sampler object.
        """
        self.batch_size = batch_size
        self.buckets = self._get_buckets(dataset)
        self.num_examples = len(dataset)
    
    def __iter__(self):
        """ An iterator is generated by randomly selecting the order
        to iterate over the buckets and then randomly iterating over
        images within that bucket. 
        
        Inputs:
            None

        Outputs:
            idxs: iterator
                The order to iterate over the examples in the dataset.
        """
        # Initialize
        batch = []

        # Iterate over buckets randomly
        dims = random.sample(list(self.buckets), len(self.buckets))
        for dim in dims:

            # Iterate over examples within buckets randomly
            bucket = self.buckets[dim]
            bucket = random.sample(bucket, len(bucket))
            for idx in bucket:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Yield half-full batch before moving to next bucket
            if len(batch) > 0:
                yield batch
                batch = []


    def __len__(self):
        """ Returns the number of examples in the dataset. """
        return self.num_examples


    def _get_buckets(self, dataset):
        """ Helper function that buckets images according to their dimensions.

        Inputs:
            dataset : torch.utils.data.Dataset object
                A PyTorch Dataset containing the examples for the dataset.

        Outputs:
            buckets : defaultdict of list
                Stores the bucketed dimensions of images as keys and
                a list of indices corresponding to those images as
                values.
        """
        buckets = defaultdict(list)
        for i in range(len(dataset)):
            img, _ = dataset[i]
            dims = img.shape
            buckets[dims].append(i)`
        return buckets

if __name__ == "__main__":
    # Sanity check
    from dataset import Im2LatexDataset
    from torch.utils.data import DataLoader

    # Use GPU if available, else use CPU
    USE_CUDA = torch.cuda.is_available()

    # Paths to datafiles
    image_path = "../data/sample/images_processed"
    formula_path = "../data/sample/formulas.norm.lst"
    train_path = "../data/sample/train_filter.lst"
    
    # Try using the sampler 
    batch_size = 20
    dataset = Im2LatexDataset(image_path, formula_path, train_path)
    sampler = Im2LatexSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=sampler, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(i, batch)

