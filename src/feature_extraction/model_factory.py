
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50
import torch
from torchvision import models

# Define path where pretrained models will be stored
MODEL_DIR = 'models'

def get_model(model_name, framework='tensorflow'):
    """
    Load and return the pretrained model.
    :param model_name: str, model to load (e.g., 'vgg19', 'resnet50')
    :param framework: str, framework to use ('tensorflow' or 'pytorch')
    :return: model
    """
    model_path = os.path.join(MODEL_DIR, f'{model_name}.h5')
    
    # Check if model file already exists in the models directory
    if framework == 'tensorflow':
        if not os.path.exists(model_path):  # Check if model is saved locally
            print(f"Model '{model_name}' not found locally. Downloading...")
            if model_name == 'vgg19':
                model = VGG19(weights='imagenet')  # Download and load VGG19
            elif model_name == 'resnet50':
                model = ResNet50(weights='imagenet')  # Download and load ResNet50
            else:
                raise ValueError(f"Model {model_name} is not supported.")
            # Save the model locally
            model.save(model_path)
            print(f"Model '{model_name}' saved to {model_path}.")
        else:
            print(f"Model '{model_name}' found locally. Loading...")
            model = tf.keras.models.load_model(model_path)  # Load the model
        return model

    elif framework == 'pytorch':
        if not os.path.exists(model_path):  # Check if model is saved locally
            print(f"Model '{model_name}' not found locally. Downloading...")
            if model_name == 'vgg19':
                model = models.vgg19(pretrained=True)  # Download and load VGG19
            elif model_name == 'resnet50':
                model = models.resnet50(pretrained=True)  # Download and load ResNet50
            else:
                raise ValueError(f"Model {model_name} is not supported.")
            # Save the model locally
            torch.save(model.state_dict(), model_path)
            print(f"Model '{model_name}' saved to {model_path}.")
        else:
            print(f"Model '{model_name}' found locally. Loading...")
            model = models.vgg19()  # Example for VGG19, modify accordingly for other models
            model.load_state_dict(torch.load(model_path))  # Load the model
        return model

    else:
        raise ValueError("Framework must be either 'tensorflow' or 'pytorch'")


'''
def get_model(model_name='vgg19'):
    if model_name == 'vgg19':
        model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'resnet50':
        model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'vit':
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model'''