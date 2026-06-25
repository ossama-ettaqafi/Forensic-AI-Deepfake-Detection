from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image)
    return tensor   # ✅ NO unsqueeze here