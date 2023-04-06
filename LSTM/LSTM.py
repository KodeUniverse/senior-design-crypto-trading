import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
## PyTorch LSTM implementation

class LSTM_model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        #Network Architecture
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        #init hidden state to zeros
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #init cell state to zeros
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        #Forward propagate the input through the network
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc2(out) #Final Output
        return out
    
    def fit(self, X_train, Y_train, learn_rate = 0.001, epochs = 50):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        ## Training

        print("Starting training loop...")
        for epoch in range(epochs):
            outputs = self.forward(X_train) # forward pass
            optimizer.zero_grad() #calc gradient, setting to zero

            #loss function calculation
            loss = criterion(outputs, Y_train)

            loss.backward() # calcualtes the errors from the loss function

            optimizer.step() #optimize using backpropagation

            #Print Loss every 50 epochs
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, loss: {loss.item():1.5f}")
