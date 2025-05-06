import torch
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from models.ssd_resnet import get_ssd_resnet_model
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

def collate_fn(batch):
    return tuple(zip(*batch))

dataset = torchvision.datasets.VOCDetection(root="./data", year="2007", image_set="train", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model = get_ssd_resnet_model(num_classes=21)  # Pascal VOC has 20 classes + background
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in dataloader:
        images = list(img.to(device) for img in images)
        targets = utils.prepare_targets(targets)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {losses.item():.4f}")
torch.save(model.state_dict(), "ssd_resnet_pascal.pth")
