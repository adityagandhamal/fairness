import numpy as np
import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import glob


print("Loading Images")
#bald_males = ["5imgs of bald males"]
#bald_females = ["5imgs of bald females"]

labels = torch.tensor([0, 0, 0, 0, 0])

bald_males = glob.glob('./Bald_Males/*.jpg')
bald_females = glob.glob('./Bald_Females/*.jpg')

#print(len(bald_males))
#print(len(bald_females))

transform = transforms.Compose(
        [transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
"""
image_set_bald_males = []
for i in range(len(bald_males)):
    image_set_bald_males.append(Image.open(bald_males[i]))

image_tensors = []
for i in range(len(image_set_bald_males)):
    image_tensors.append(transform(image_set_bald_males[i]))

for i in range(len(image_tensors)):
    image_tensors[i] = torch.unsqueeze(image_tensors[i], 0)

#print(len(image_set_bald_males))
#print(image_set_bald_males)

#print(len(image_tensors))
print("Images Loaded Sucessfully")

# Load Model

print("Loading Model Architecture")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

in_features = model.fc.in_features
out_features = 1

class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

model.fc = Classifier(in_features=in_features, out_features=out_features)
#print(model)
print("Model Architecture Loaded")

# Load model checkpoints
print("Loading Model Checkpoints")
checkpoint_path = 'adversarial_classifier_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# Load model state dict
model.load_state_dict(checkpoint['model_state_dict'])
print("Model Checkpoints Loaded Successfully")


model.eval()
correct = 0
for i in range(len(image_tensors)):
    output = model(image_tensors[i])
    _, predicted = torch.max(output, 1)
    #break
    correct += (predicted == labels[i]).sum().item()
    #print(predicted == labels[i])
    #print(correct)
    #break
acc = correct/(len(image_tensors))
print("Accuracy:", acc*100)    

#print(output)
#print(predicted)
"""

image_set_bald_females = []
for i in range(len(bald_females)):
    image_set_bald_females.append(Image.open(bald_females[i]))

image_tensors = []
for i in range(len(image_set_bald_females)):
    image_tensors.append(transform(image_set_bald_females[i]))

for i in range(len(image_tensors)):
    image_tensors[i] = torch.unsqueeze(image_tensors[i], 0)

#print(len(image_set_bald_males))
#print(image_set_bald_males)

#print(len(image_tensors))
print("Images Loaded Sucessfully")

# Load Model

print("Loading Model Architecture")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

in_features = model.fc.in_features
out_features = 1

class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

model.fc = Classifier(in_features=in_features, out_features=out_features)
#print(model)
print("Model Architecture Loaded")

# Load model checkpoints
print("Loading Model Checkpoints")
checkpoint_path = 'adversarial_classifier_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# Load model state dict
model.load_state_dict(checkpoint['model_state_dict'])
print("Model Checkpoints Loaded Successfully")


model.eval()
correct = 0
for i in range(len(image_tensors)):
    output = model(image_tensors[i])
    _, predicted = torch.max(output, 1)
    #break
    correct += (predicted == labels[i]).sum().item()
    #print(predicted == labels[i])
    #print(correct)
    #break
acc = correct/(len(image_tensors))
print("Accuracy:", acc*100)    

#print(output)
#print(predicted)