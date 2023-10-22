from cog import BasePredictor, Input, Path
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO

class Predictor(BasePredictor):
    def setup(self):
        """Load the model"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RoadDetection().to(self.device)
        self.model.load_state_dict(torch.load("road_detection_model.pth", map_location=self.device))
        self.model.eval()  # Important if your model has different behavior during training vs inference

    def predict(self, image: Path) -> Path:
        """Run a single prediction on the model"""
        input_data = self.preprocess_image(image)
        with torch.no_grad():
            output = self.model(input_data)
        output_image = self.postprocess_output(output)

        # Save output_image to a file
        output_path = Path("/tmp/output_image.jpg")  # Or another appropriate temporary path
        output_image.save(output_path, format="JPEG")

        return output_path

    def preprocess_image(self, image_path: Path):
        """Preprocess an image file to the format expected by the model"""
        image = Image.open(image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        input_data = transform(image).unsqueeze(0)  # Add batch dimension
        input_data = input_data.to(self.device)
        return input_data

    def postprocess_output(self, output):
        """Convert the model output to a PIL Image"""
        output_mask = output.squeeze().cpu().numpy()
        pil_image = Image.fromarray((output_mask * 255).astype('uint8'))
        return pil_image

class RoadDetection(nn.Module):
    def __init__(self):
        super(RoadDetection, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu5 = nn.ReLU()
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.deconv1(x)
        x = self.relu4(x)
        
        x = self.deconv2(x)
        x = self.relu5(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x