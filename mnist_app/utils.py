import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # first hidden layer: 784 x 256
        self.hidden_0 = nn.Linear(784, 128)
        self.hidden_1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        
    def forward(self, x):
        # flatten input to vector of 764
        x = x.view(x.shape[0], -1)
        # forward pass
        x = self.hidden_0(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        
        x = self.softmax(x)
        
        return x

model = Network()

model.load_state_dict(torch.load('/Users/nabanibanik/Desktop/mnist_app/MNIST_model.pth', map_location=torch.device('cpu')))
model.eval()


def load_mnist_images():
    """
    Loads MNIST images from the 'mnist_images' directory and returns them as a list of arrays.
    """
    mnist_images = []
    mnist_dir = "mnist_images"
    for file_name in os.listdir(mnist_dir):
        if file_name.endswith('.jpg'):
            img_path = os.path.join(mnist_dir, file_name)
            image = np.array(Image.open(img_path).convert('L').resize((28, 28)))
            op = model(torch.from_numpy(image/255.0).view(1,-1).float()).detach().numpy()


            # mnist_images.append((image, img_path))
            mnist_images.append((op, image,  img_path))
    return mnist_images

def find_nearest_images(user_image, mnist_images, top_n=5):
    """
    Finds the top_n nearest MNIST images to the user_image.
    """
    # Flatten the user image
    # user_image_flat = user_image.flatten().reshape(1, -1)

    user_image_flat = model(torch.from_numpy(user_image/255.0).view(1,-1).float()).detach().numpy()
    
    
    # Flatten each MNIST image
    mnist_images_flat = np.array([img.flatten() for img, _ ,_ in mnist_images])
    
    # Calculate Euclidean distances
    distances = euclidean_distances(user_image_flat, mnist_images_flat).flatten()
    
    # Get the indices of the nearest images
    nearest_indices = np.argsort(distances)[:top_n]
    
    # Return the nearest images along with their distances
    nearest_images = [(mnist_images[i][1], distances[i]) for i in nearest_indices]
    return nearest_images
