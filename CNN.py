import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
import sys
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions
import torch.optim as optim # Optimizers


#Hyperparameter
BATCH_SIZE = 320
global LEARNING_RATE


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=1) # First Conv Layer
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)  # For pooling
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0) # First Conv Layer
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2) 
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(400, 120)  # First FC HL
       
        self.fc2= nn.Linear(120, 84) # Output layer
      
        self.fc3= nn.Linear(84, 10)

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.bn1(self.conv1(x))) # Shape: (B, 5, 28, 28)
      x = self.pool1(x)  # Shape: (B, 5, 14, 14)
      x = F.relu(self.bn2(self.conv2(x)))
      x = self.pool2(x)
      x = self.flatten(x) # Shape: (B, 980)
      x = F.tanh(self.fc1(x))# Shape (B, 256)
      x = F.tanh(self.fc2(x))
      x = self.fc3(x)  # Shape: (B, 10)
      return x  
    
# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# Create the transform sequence
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
])

# Define the training and testing functions

def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)

def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

def main():
    operation = sys.argv[1]
    #Load Data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False)
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if (operation == "-load"):
        print("loading params...")
        try:
            print("loading params...")
            checkpoint = torch.load('cnn_test_accuracy.pth')
            test_accuracy = checkpoint['test_accuracy']
            print(test_accuracy)
            
        except FileNotFoundError:
            print ("No test accuracy saved .First train data by using -save operation.")

    elif (operation == "-save"):
        cnn = CNN().to(device)
        LEARNING_RATE=0.1
        MOMENTUM = 0.9


        # Define the loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
        optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        test_acc=0.0
        best_test_acc=0
        no_improvement_count=0

        # Train the CNN for 10 epochs
        for epoch in range(15):
            # check if test accuracy has improved atleast by 0.01
            if test_acc > best_test_acc+0.01:
                best_test_acc = test_acc
                no_improvement_count = 0
            else:
                
                no_improvement_count += 1
            #there's some improvement
            if no_improvement_count>=2 :
                LEARNING_RATE*=0.5
                MOMENTUM=0.95

            
   
    
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
            train_loss = train(cnn, train_loader, criterion, optimizer, device)
            test_acc = test(cnn, test_loader, device)
            print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")

        checkpoint = torch.load('cnn_test_accuracy.pth')
        best_test_acc = checkpoint['test_accuracy']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ("Usage: python paging.py [number of page frames]")
    else:
        main()
