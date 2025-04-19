

from src.data_processing.dataset_loader import load_dataset
#from src.feature_extraction.feature_extractor import extract_features
from src.feature_extraction.model_factory import get_model
#from src.search.search_engine import SearchEngine

#dataset = load_dataset('cifar10')

# Step 2: Choose a model (e.g., VGG19, ViT)
model = get_model('vgg19')
