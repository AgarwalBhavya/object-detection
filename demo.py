import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
from models.ssd_resnet import get_ssd_resnet_model
import matplotlib.pyplot as plt
import torchvision
import matplotlib
matplotlib.use('TkAgg')


model = get_ssd_resnet_model(num_classes=21)
model.load_state_dict(torch.load("ssd_resnet_pascal.pth"))
model.eval()

transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
image = Image.open("sample.jpg")
img = transform(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(img)[0]

boxes = prediction["boxes"]
labels = prediction["labels"]
scores = prediction["scores"]

drawn = torchvision.utils.draw_bounding_boxes((img[0] * 255).byte(), boxes, labels=[str(l.item()) for l in labels])
plt.imshow(drawn.permute(1, 2, 0))
plt.axis("off")
plt.show()
