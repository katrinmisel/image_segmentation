# Street image segmentation for autonomous vehicles

In this project, I will be using the Cityscapes Dataset: https://www.cityscapes-dataset.com/

The goal of the project is to train a Keras model on our dataset to detect various objects in the image and to be able to segment an image into the following 8 categories: flat, human, vehicle, construction, object, nature, sky and void.


<img src="https://user-images.githubusercontent.com/37838896/202401774-1bc883e6-50bc-4eee-a03a-e78d79d173fa.png" width="500"/> 
The dataset contains images taken in 50 cities, over several months, at daytime with good/medium weather conditions. 


We will be testing data augmentation to increase the size of our training dataset and improve the performance of our model. We will also be testing several different metrics and pre-trained models.


The final model will be deployed via an API.
