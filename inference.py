import torch
import torchvision
from PIL import Image
from model import get_model

def main():
    model = get_model()
    model.load_state_dict(torch.load('model.pth'))
    
    image = Image.open('4.png')
    
    image = torchvision.transforms.ToTensor()(image)
    
    # to black and white
    image = torch.mean(image, dim=0, keepdim=True)
        
    flattened_image = torch.nn.Flatten()(image)
    
    output = model(flattened_image)
    
    print(output.shape)
    
    print(output)

if __name__ == "__main__":
    main()