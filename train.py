import torch 
import torchvision
import torchvision.datasets.mnist as mnist 

from model import get_model

def main():
    dataset = mnist.MNIST('./data', download=True)
    
    model = get_model()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.MSELoss()
    
    total_loss = 0
    proper_answers = 0
    
    for i, (image, label) in enumerate(dataset):
        image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(label)
        
        optimizer.zero_grad()
        
        flattened_image = torch.nn.Flatten()(image)
        output = model(flattened_image)
        
        desired = torch.zeros(10)
        desired[label] = 1
        
        loss = loss_func(output, desired)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        answer = torch.argmax(output).item()

        if answer == label:
            proper_answers += 1
        

        if i % 100 == 0:
            print(f'Loss: {total_loss / (i + 1)}')
            print(f'Accuracy: {proper_answers / (i+1)}')

    torch.save(model.state_dict(), 'model.pth')
    
if __name__ == '__main__':
    main()
