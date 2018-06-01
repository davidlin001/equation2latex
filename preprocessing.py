# Title: preprocess.py
# Author: Cody Kala
# Date: 5/31/2018
# =====================
# This module defines functions to preprocess the im2latex-100k dataset.

def read_formulas(formula_file):
    """ Reads in the formulas from |formula_file|. 
    
    Inputs:
        formula_file : string
            The name of the file where the mathematical formulas are saved.

    Outputs:
        formulas : dict
            A dictionary whose keys are formula IDs (i.e. the row index in
            |formula_file| where that formula is located) and the values
            are the formulas (as strings).
        bad_rows : list
            List of row indices where a UnicodeDecodeError was encountered.
    """
    formulas = {}
    bad_rows = []
    count = 0
    with open(formula_file, mode="rb") as f:
        for i, bline in enumerate(f):
            count += 1
            try:
                formula = bline.decode("utf-8")
                formulas[i] = formula
            except UnicodeDecodeError:
                bad_rows.append(i)
    print(count)
    return formulas


def read_lookup(lookup_file):
    """ Reads the |lookup_file| to associate formulas to images.

    Inputs:
        lookup_file : string
            The name of the file where the lookup information that relates
            the formula IDs to image names is saved.

    Outputs:
        idx_to_image : dict
            A dictionary whose keys are formula IDs (i.e. the row index in
            |formula_file| where that formula is located) and the name of
            the image file for the formula.
        bad_rows : list
            List of indices where a UnicodeDecodeError was encountered.
    """
    idx_to_image = {}
    bad_rows = []
    with open(lookup_file, mode="rb") as f:
        for i, bline in enumerate(f):
            try:
                line = bline.decode("utf-8")
                idx, img_name, render_type = line.strip().split()
                idx_to_image[int(idx)] = img_name
            except UnicodeDecodeError:
                bad_rows.append(i)
    print(len(bad_rows))
    return idx_to_image


def clean_formulas_and_lookup(formulas, idx_to_image):
    """ Creates new .lst files for the formulas and lookup info, removing
    row numbers for formulas that are not in |formulas| due to a
    UnicodeDecodeError. 

    Inputs:
        formulas : dict
            A dictionary whose keys are row indices into the original
            |formula_file| and whose values are the formulas (strings).
        idx_to_image : dict
            A dictionary whose keys are row indices into the original
            |formula_file| and whose values are the image names (strings).

    Outputs:
        new_formulas : dict
            A dictionary whose keys are row indices into a new formula file
            and whose values are the formulas. The new formula file does not
            include lines for formulas that encountered a UnicodeDecodeError.
        new_idx_to_image : dict
            A dictionary whose keys are row indices into the new formula file
            and whose values are the image names (strings).
    """
    # Initialize 
    new_formulas = {}
    new_idx_to_image = {}

    # Reassign indices
    for new_idx, idx in enumerate(sorted(idx_to_image)):
        if idx not in formulas:
            continue
        new_formulas[new_idx] = formulas[idx]
        new_idx_to_image[new_idx] = idx_to_image[idx]

    return new_formulas, new_idx_to_image


def save_data(formulas, idx_to_img, new_formula_path, new_lookup_path):
    """ Saves the cleaned data in the same format as it was read in, minus 
    the rendering type. 
    
    Inputs:
        formulas : dict
            A dictionary whose keys are row indices into the file where
            the formulas are stored and whose values are the formulas.
        idx_to_img : dict
            A dictionary whose keys are row indices into the file where
            the formulas are stored and whose values are the image names.
        new_formula_path : string
            The path to the file where the "cleaned" formulas are saved.
        new_lookup_path : string
            The path to the file where the "cleaned" lookup information
            is saved.

    Note: "clean" here refers to the data not throwing any UnicodeDecodeErrors.
    """
    with open(new_formula_path, "w") as f, open(new_lookup_path, "w") as g:
        for idx in sorted(idx_to_img):
            f.write("{}".format(formulas[idx]))
            g.write("{} {}.png\n".format(idx, idx_to_img[idx]))
            

if __name__ == "__main__":

    # Load the data
    formulas = read_formulas("im2latex/original/im2latex_formulas.lst")
    train_idx_to_img = read_lookup("im2latex/original/im2latex_train.lst")
    val_idx_to_img = read_lookup("im2latex/original/im2latex_validate.lst")
    test_idx_to_img = read_lookup("im2latex/original/im2latex_test.lst")

    print("Sanity check")
    print("Formula count: {}".format(len(formulas)))
    print("Lookup count: {}".format(len(train_idx_to_img) + len(val_idx_to_img) + len(test_idx_to_img)))
    print("Difference: {}".format(len(formulas) - len(train_idx_to_img) - len(val_idx_to_img) - len(test_idx_to_img)))

    # Clean data
    train_formulas, new_train_idx_to_img = clean_formulas_and_lookup(formulas, train_idx_to_img)
    val_formulas, new_val_idx_to_img = clean_formulas_and_lookup(formulas, val_idx_to_img)
    test_formulas, new_test_idx_to_img = clean_formulas_and_lookup(formulas, test_idx_to_img)

    print(len(train_formulas))
    print(len(val_formulas))
    print(len(test_formulas))

    assert(len(train_formulas) == len(new_train_idx_to_img))
    assert(len(val_formulas) == len(new_val_idx_to_img))
    assert(len(test_formulas) == len(new_test_idx_to_img))

    # Save cleaned data
    save_data(train_formulas, new_train_idx_to_img, "im2latex/train_formulas.lst", "im2latex/train_lookup.lst")
    save_data(val_formulas, new_val_idx_to_img, "im2latex/val_formulas.lst", "im2latex/val_lookup.lst")
    save_data(test_formulas, new_test_idx_to_img, "im2latex/test_formulas.lst", "im2latex/test_lookup.lst")


