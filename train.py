import torch 
import torchvision
import torchvision.datasets.mnist as mnist 

from model import get_model

def train_model(model, train_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.MSELoss()
    
    total_loss = 0    
    
    for i, (image, label) in enumerate(train_dataset):
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
            
        if i % 100 == 0:
            print(f'Loss: {total_loss / (i + 1)}')
    
def evaluate_model(model, test_dataset):
    correct = 0
    total = 0
    
    for i, (image, label) in enumerate(test_dataset):
        image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(label)
        
        flattened_image = torch.nn.Flatten()(image)
        output = model(flattened_image)
        
        prediction = torch.argmax(output)
        
        if prediction == label:
            correct += 1
        
        total += 1
        
    print(f'Accuracy: {correct / total}')

def main():
    train_dataset = mnist.MNIST('./data_train', train=True, download=True)
    test_dataset = mnist.MNIST('./data_test', train=False, download=True)
    
    model = get_model()
    train_model(model, train_dataset)
    evaluate_model(model, test_dataset)

    torch.save(model.state_dict(), 'model.pth')
    
if __name__ == '__main__':
    main()
