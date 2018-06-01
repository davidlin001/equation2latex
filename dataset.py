# Title: dataset.py
# Author: Cody Kala
# Date: 5/31/2018
# ====================
# Module implements a Dataset class for the im2latex-100k dataset for more
# efficient lookup during training.

import torch
import torchvision
import os.path
from PIL import Image
from torch.utils import data


class Im2LatexDataset(data.Dataset):
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
        self.images_path = images_path
        self.formulas = self._read_formulas(formulas_path)
        self.idx_to_img = self._read_lookup(lookup_path)
        assert(len(self.formulas) == len(self.idx_to_img))

    def __len__(self):
        """ Returns the number of examples in the dataset. """
        return len(self.formulas)


    def __getitem__(self, idx):
        """ Returns the image (and label, if present) corresponding to
        the index |idx|. 
        
        Inputs:
            idx : int
                Index between 0 and len(dataset) - 1, inclusive that refers to
                an image (and label) in the dataset.
                
        Outputs:
            image : torch.Tensor
                The image file corresponding to the given |idx|.
            formula : string
                The formula corresponding to the given |idx|.
        """
        formula = self.formulas[idx]
        image_name = self.idx_to_img[idx]
        image_path = os.path.join(self.images_path, image_name)
        image = Image.open(image_path)
        image = torchvision.transforms.functional.to_tensor(image)
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
            formulas = { i : line.strip() for i, line in enumerate(lines) }
            return formulas


    def _read_lookup(self, lookup_path):
        """ Reads in the lookup information from the file at |lookup_path| 
        that maps formulas to images.

        Inputs:
            lookup_path : string
                The path to the file that contains the lookup information.


                A dictionary whose keys are row indices into |formula_path|
                and whose values are image names.
        """
        with open(lookup_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            idxs = [int(line[0]) for line in lines]
            img_names = [line[1] for line in lines]
            idx_to_img = {idx : img_name for idx, img_name in zip(idxs, img_names)}
            return idx_to_img

if __name__ == "__main__":
    # Sanity checks
    # Train set
    images_path = "im2latex/images"
    formulas_path = "im2latex/train_formulas.lst"
    lookup_path = "im2latex/train_lookup.lst"
    train_dataset = Im2LatexDataset(images_path, formulas_path, lookup_path)
    assert(len(train_dataset) == 83859) 

    # Val set
    formulas_path = "im2latex/val_formulas.lst"
    lookup_path = "im2latex/val_lookup.lst"
    val_dataset = Im2LatexDataset(images_path, formulas_path, lookup_path)
    assert(len(val_dataset) == 9317)

    # Test set
    formulas_path = "im2latex/test_formulas.lst"
    lookup_path = "im2latex/test_lookup.lst"
    test_dataset = Im2LatexDataset(images_path, formulas_path, lookup_path)
    assert(len(test_dataset) == 10352) 
