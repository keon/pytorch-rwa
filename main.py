import time
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RWA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RWA, self).__init__()

        self.max_steps = 1
        self.batch_size = 1
        self.hidden_size = hidden_size

        self.n = Variable(torch.Tensor(self.batch_size, hidden_size), requires_grad=True)
        self.d = Variable(torch.Tensor(self.batch_size, hidden_size), requires_grad=True)

        self.x2u = nn.Linear(input_size, hidden_size)
        self.c2g = nn.Linear(input_size + hidden_size, hidden_size)
        self.c2q = nn.Linear(input_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        h = F.tanh(hidden)

        for i in range(len(input)):
            combined = torch.cat((input[i], h), 1)
            u = self.x2u(input[i])
            g = self.c2g(combined)
            q = self.c2q(combined)
            q_greater = F.relu(q)
            scale = torch.exp(-q_greater)
            a_scale = torch.exp(q-q_greater)
            self.n = (self.n * scale) + ((u * F.tanh(g)) * a_scale)
            self.d = (self.d * scale) + a_scale
            h = F.tanh(torch.div(self.n, self.d))
        output = self.out(h)
        return output, h

    def init_hidden(self):
        return Variable(torch.randn(1, self.hidden_size))

n_hidden = 128
rwa = RWA(n_letters, n_hidden, n_categories)
print("n_letters:", n_letters, "n_hidden:", n_hidden, "n_categories:", n_categories)
print(rwa)


input = Variable(line2tensor('Keon'))
hidden = rwa.init_hidden()

output, next_hidden = rwa(input, hidden)
print(output, next_hidden)

learning_rate = 0.005
def train (categroy_tensor, line_tensor):
    hidden = rwa.init_hidden()
    hidden = Variable(hidden.data)
    rwa.zero_grad()
    output, hidden = rwa(line_tensor, hidden)
    loss = criterion(output, category_tensor)
    print("loss:" , loss)
    loss.backward()

    for p in rwa.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.data[0]

if __name__ == "__main__":
    n_epochs = 100000
    print_every = 5000
    plot_every = 1000

    rwa = RWA(n_letters, n_hidden, n_categories)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = training_pair()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
