import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS,N_LETTERS
from utils import load_data,letter_to_tensor,letter_to_index,random_training_example
print("Torch.nn")
# class RNN(nn.Module):
#     def __init__(self,input_size,hiddne_size,output_size):
#         super(RNN,self).__init__()
#         self.hidden_size=hiddne_size
#         self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
#         self.i2o=nn.Linear(input_size+hidden_size,output_size)
#         self.softmax=nn.LogSoftmax(dim=1)
#     def forward(self,input_tensor,hidden_tensor):
#         combined=torch.cat((input_tensor,hidden_tensor),1)
#         hidden=self.i2h(combined)
#         output=selc.softmax(self.i2o(combined))
#         return output,hidden
#     def init__hidden(self):
#         return torch.zeros((1,self.hidden_size))

class RNN(nn.Module):
    def __init__(self,inp,hid1,hid2,op):
        super(RNN,self).__init__()        
        self.hid1=hid1
        self.hid2=hid2
        self.fc1=nn.Linear(inp+hid1,hid1)
        self.fc2=nn.Linear(hid1+hid2,hid2)
        self.fc3=nn.Linear(hid2,op)
        self.log_softmax=nn.LogSoftmax(dim=1)
    def forward(self,input,hidden1,hidden2):
        combined=torch.cat((input,hidden1),1)
        hid1=self.fc1(combined)
        combined=torch.cat((hid1,hidden2),1)
        hid2=self.fc2(combined)
        op=self.fc3(hid2)
        op=self.log_softmax(op)
        return op,hid1,hid2
	
    def hidden_init(self,f):
        return torch.zeros((1,self.hid1)) if f==1 else torch.zeros((1,self.hid2))

category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden_1 = 128
n_hidden_2 = 64
rnn = RNN(N_LETTERS, n_hidden_1,n_hidden_2, n_categories)
criterion=nn.NLLLoss()
learning_rate=0.05
optimizer=torch.optim.SGD(rnn.parameters(),lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden1, hidden2 = rnn.hidden_init(1).to(device), rnn.hidden_init(2).to(device) 
def train(line_tensor,category_tensor,hidden1,hidden2):#,hidden1,hidden2
    # hidden1=rnn.hidden_init(1)
    # hidden2=rnn.hidden_init(2)
    # hidden1, hidden2 = rnn.hidden_init(1).to(device), rnn.hidden_init(2).to(device)  # Move hidden states to device
    for i in range(line_tensor.size()[0]):
        output,hidden1,hidden2=rnn.forward(line_tensor[i],hidden1,hidden2)
         # Detach hidden states to prevent tracking history from previous steps
        hidden1, hidden2 = hidden1.detach(), hidden2.detach()
    loss=criterion(output,category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(loss.item())
    return output,loss.item(),hidden1,hidden2

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

current_loss=0
all_losses=[]
plot_steps, print_steps = 100, 50
n_iters = 10000
# hidden1=rnn.hidden_init(1)
# hidden2=rnn.hidden_init(2)
for i in range(n_iters):
    category,line,category_tensor,line_tensor=random_training_example(category_lines,all_categories)
    
    output,loss,hidden1,hidden2=train(line_tensor,category_tensor,hidden1,hidden2)#,hidden1,hidden2
    current_loss+=loss

    if (i+1)%plot_steps==0:
        all_losses.append(current_loss/plot_steps)
        current_loss=0

    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")
        
    
plt.figure()
plt.plot(all_losses)
plt.xlabel("Number of iterations for plot")
plt.ylabel("Loss")
plt.legend()
plt.show()