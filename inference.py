import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mnist= datasets.MNIST(root='data', train=False, transform=transform, download=True)
loader = torch.utils.data.DataLoader(mnist, batch_size=32, shuffle=False)

#model.fc = torch.nn.Linear(model.fc.in_features, 15)
model.eval().to(device)


num=0
ans=0

with torch.no_grad():
    for image, label in loader:
        image, label = image.to(device), label.to(device)
        outputs = model(image)

        _, answer= torch.max(outputs, 1)

        num+= label.size(0)
        temp= (answer == label).sum()
        ans=temp.item()

a_r=ans/num
print("Accuracy is:",a_r)




