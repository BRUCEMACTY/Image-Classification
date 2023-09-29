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


# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(32*32*3,2048)  # Input is image with shape (28x28)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)  # First HL
        self.fc3= nn.Linear(1024, 512) # Second HL
  

        self.fc4= nn.Linear(512, 256) #Thi
        self.fc5= nn.Linear(256, 128) # Forth HL
        self.dropout=nn.Dropout(p=0.07)
        self.fc6= nn.Linear(128, 10) # Forth HL4

       # self.dropout=nn.Dropout(p=0.7)

      #  self.fc7= nn.Linear(50, 10) 
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = F.relu(self.fc1(x))  # First Hidden Layer
      x = F.relu(self.fc2(x))  # Second Hidden Layer
      x = F.relu(self.fc3(x))  # Third Hidden Layer
      x = F.relu(self.fc4(x))  # Second Hidden Layer   
      x = F.relu(self.fc5(x)) 
      #x = F.relu(self.fc6(x)) 
      x = self.fc6(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)
    
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
            checkpoint = torch.load('mlp_test_accuracy.pth')
            test_accuracy = checkpoint['test_accuracy']
            print(test_accuracy)
            print("loading params...")
        except FileNotFoundError:
            print ("No test accuracy saved .First train data by using -save operation.")

    elif (operation == "-save"):
        mlp = MLP().to(device)

        LEARNING_RATE = 0.04
        MOMENTUM = 0.9


        # Train the MLP for 5 epochs
        for epoch in range(15):
            if epoch >6:
                LEARNING_RATE *=0.1
    # Define the loss function, optimizer, and learning rate scheduler
            criterion = nn.NLLLoss()
            optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

            train_loss = train(mlp, train_loader, criterion, optimizer, device)
            test_acc = test(mlp, test_loader, device)
            print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")

        checkpoint = torch.load('mlp_test_accuracy.pth')
        best_test_acc = checkpoint['test_accuracy']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ("Usage: python MLP.py -[OPERATION : -load/-save]")
    else:
        main()
