import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms
from models.ssd_resnet import get_ssd_resnet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = VOCDetection(root="./data", year="2007", image_set="val", download=True, transform=transform)

model = get_ssd_resnet_model(num_classes=21)
model.load_state_dict(torch.load("ssd_resnet_pascal.pth"))
model.eval()
model = model.to(device)

total = 0
for img, _ in dataset:
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img)[0]
    print(prediction)
    total += 1
    if total == 5:
        break
