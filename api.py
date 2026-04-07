from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model import MNISTNet

app = FastAPI()

# Load model
model = MNISTNet()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing matching MNIST training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess and Infer
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()
        confidence = torch.exp(output).max().item()

    return {"prediction": prediction, "confidence": f"{confidence*100:.2f}%"}