import urllib.request
from keras.models import load_model

# URL of the h5 model file on Google Drive
url = 'https://drive.google.com/file/d/1rLXBf-QIX36nFGTo6CL-9EcGQk5dRHhP/view?usp=share_link'

# Download the file and save it locally
urllib.request.urlretrieve(url, 'model.h5')

# Load the model from the local file
model = load_model('model.h5')
