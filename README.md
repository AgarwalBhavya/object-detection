# object-detection
## Experience Report – Object Detection Assignment

### Challenges I Faced
- Integrating SSD with a custom ResNet backbone required adjusting feature map sizes.
- Getting Pascal VOC labels in the right format for training took time.

### How I Used AI Tools
- I used ChatGPT and Stack Overflow to understand how SSD layers are structured.
- AI tools helped me debug PyTorch dimension mismatches faster.

### What I Learned
- How SSD works internally: anchor generation, classification and localization heads.
- Importance of balanced dataset and loss debugging.

### What Surprised Me
- Even pretrained backbones need a lot of tweaking when added to new heads.
- Getting decent mAP takes more than just architecture—it needs good data handling.

### Writing Code Myself vs AI
- AI tools speed up boilerplate and understanding unfamiliar libraries.
- Writing training/eval logic myself helped me internalize model behavior better.

### Suggestions to Improve the Assignment
- Include label formatting guidelines for beginners.
- Recommend using torchvision detection models as base for rapid prototyping.
