import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from train import generate_vocab_mapping
sys.path.insert(0, "./models/im2markup/")
from utils.dataset import Im2LatexDataset
from models.im2latex.model import Im2Latex

MAX_LENGTH = 50
EMBED_SIZE = 80

def preprocess(img, size=224):
    transform = T.Compose([
        # Check resize
        T.Resize(size),
        T.ToTensor()#,
        #T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
        #            std=SQUEEZENET_STD.tolist()),
        #T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: 1 Input Image; Tensor of shape (N, 3, H, W). N Here is the max length of
    predicted equations
    - y: Grouth truth equation given as list of tokens.  LongTensor of shape (N,)
    - model: Model used for prediction

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    scores, indices = model(X)
    loss = 0.0

    for i in range(max_length):
        scores_i = scores[i,:]
        targets_i = [token_to_index.get(s, 3) for s in np.array(y)[i]]
        targets_i = torch.LongTensor(targets_i)
        loss += torch.sum(loss_fn(scores_i,targets_i))

    loss.backward()
    grads = torch.abs(X.grad)
    saliency, indices = torch.max(grads, dim = 1)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def show_saliency_maps(X, y, model):
    # Convert X and y from numpy arrays to Torch Tensors
    #X_tensor = preprocess(Image.fromarray(X))
    X_tensor = preprocess(X)

    y_tensor = torch.CharTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    plt.subplot(2, 1, 1)
    plt.imshow(X)
    plt.axis('off')
    plt.title("Saliency Example")
    plt.subplot(2, 1, 2)
    plt.imshow(saliency, cmap=plt.cm.hot)
    plt.axis('off')
    plt.gcf().set_size_inches(12, 5)
    plt.show()

if __name__ == "__main__":
    images_path = "../data/sample/images_processed/"
    formulas_path = "../data/sample/formulas.norm.lst"
    train_path = "../data/sample/train_filter.lst"
    vocab_path = "../data/sample/latex_vocab.txt"

    dataset = Im2LatexDataset(images_path, formulas_path, train_path)
    index_to_token, token_to_index = generate_vocab_mapping(vocab_path)
    X = Image.open(images_path + dataset.idx_to_img[0])
    equation = dataset.idx_to_formula[0]
    
    print(equation)
    y = [] 
    for token in equation: 
        y.append(token_to_index[token])
    
    print(y)
    decoder_config = {"vocab_size": len(index_to_token), "max_length": MAX_LENGTH, "embed_size": EMBED_SIZE}
    model = Im2Latex(decoder_config)


    show_saliency_maps(X, y, model)
    
        
