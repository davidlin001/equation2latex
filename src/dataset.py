# Title: dataset.py
# Author: Cody Kala
# Date: 5/31/2018
# ====================
# Module implements a Dataset class for the im2latex-100k dataset for more
# efficient lookup during training.

import torch
import os.path
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Im2LatexDataset(Dataset):
    """ Dataset class that accesses the images and gold labels for the 
    im2latex-100k dataset. 
    
    Note: The im2latex-100k training, validation, and test sets are
    all labeled.
    """
    
    def __init__(self, images_path, formulas_path, lookup_path):
        """ Initializes the Dataset object.

        Inputs:
            images_path : string
                The filepath to the directory containing the images.
            formulas_path : string
                The filepath to the file containing the formulas.
            lookup_path : string
                The filepath containing the information that maps
                images to formulas. Each line has the following format:

                    formula_idx image_name

                where 

                    formula_idx : int 
                        The line number where formula is in |formula_path|.
                    image_name : string
                        The name of the image in |images_path|.

        Outputs:
            None
        """
        # Store filepaths for later
        self.images_path = images_path
        self.formulas_path = formulas_path
        self.lookup_path = lookup_path
        
        # Generate lookup dictionaries for indexing
        self.idx_to_formula = self._read_formulas(formulas_path)
        self.idx_to_img = self._read_lookup(lookup_path)

        # Reindex so all indices between [0, len(dataset)-1]
        self._reindex_examples()


    def __len__(self):
        """ Returns the number of examples in the dataset. """
        return len(self.idx_to_formula)


    def __getitem__(self, idx):
        """ Returns the image (and label, if present) corresponding to
        the index |idx|. 
        
        Inputs:
            idx : int
                Index between 0 and len(dataset) - 1, inclusive that refers to
                an image (and formula) in the dataset.
                
        Outputs:
            image : torch.Tensor
                The image file corresponding to the given |idx|.

        """
        formula = self.idx_to_formula[idx]
        image_name = self.idx_to_img[idx]
        image_path = os.path.join(self.images_path, image_name)
        image = Image.open(image_path)
        image = transforms.functional.to_tensor(image)
        return image, formula


    def _read_formulas(self, formula_path):
        """ Reads in the formulas from the file at |formula_path|.

        Inputs:
            formula_path : string
                The path to the file that contains the LaTeX code.

        Outputs:
            formulas : dict
                 A dictionary whose keys are row indices into |formula_path|
                 and whose values are LaTeX codes.
        """
        with open(formula_path, "r") as f:
            lines = f.readlines()
            idx_to_formula = {idx : line.strip() for idx, line in enumerate(lines)}
            return idx_to_formula


    def _read_lookup(self, lookup_path):
        """ Reads in the lookup information from the file at |lookup_path| 
        that maps formulas to images.

        Inputs:
            lookup_path : string
                The path to the file that contains the lookup information.


                A dictionary whose keys are row indices into |formula_path|
                and whose values are image names
.        """
        with open(lookup_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            img_names = [line[0] for line in lines]
            idxs = [int(line[1]) for line in lines]
            idx_to_img = {idx : img_name for idx, img_name in zip(idxs, img_names)}
            return idx_to_img

    def _reindex_examples(self):
        """ Helper function that maps examples to indices in the range 
        [0, len(dataset)-1].
    
        Inputs:
            None

        Outputs:
            None, but modifies |self.idx_to_formulas| and |self.idx_to_img|.
        """
        # Initialize
        new_idx_to_formula = dict()
        new_idx_to_img = dict()
        new_idx = 0

        # Loop over examples
        for idx in sorted(self.idx_to_img):
            new_idx_to_formula[new_idx] = self.idx_to_formula[idx]
            new_idx_to_img[new_idx] = self.idx_to_img[idx]
            new_idx += 1

        # Update lookup dicts
        self.idx_to_formula = new_idx_to_formula
        self.idx_to_img = new_idx_to_img

if __name__ == "__main__":
    # Sanity check
    image_path = "../data/full/images_processed/"
    formula_path = "../data/full/im2latex_formulas.norm.lst"



    # Example iteration using Dataloader
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for i, (batch_images, batch_idxs) in enumerate(dataloader):
        print(batch_images.size())
