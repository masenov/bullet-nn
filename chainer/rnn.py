import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable, optimizers, optimizer
from chainer import cuda
#import debug

class SimpleRNN0(Chain):
    def __init__(self, dim_input, n_hidden):
        super().__init__()

        # Input to hidden unit
        self.add_link('i2h',L.Linear(dim_input,n_hidden, nobias=True))

        # Hidden to hidden unit
        self.add_link('h2h',L.Linear(n_hidden,n_hidden, nobias=True))

        # Hidden unit to output
        self.add_link('h2y',L.Linear(n_hidden,dim_input, nobias=True))


        # Recurrent hidden state (or memory)
        self.mem = Variable(np.ones((1,3),dtype=np.float32))
        self.mem.to_gpu()

    def __call__(self,x):
        self.to_gpu()
        # Process new input
        hx = self.i2h(x)

        # Process stored memory
        hh = self.h2h(self.mem)

        # Calculate a new memory state while considering
        # the old memory and the new input
        h =  F.tanh(hx + hh)

        # Replace the old memory state with the latest one.
        self.mem = h

        # Process and return an output given the new memory state
        return self.h2y(h)



# Our training function
def train_SimpleRNN0(model, nb_epochs = 1, backprop_timer = 10):
    accumulated_loss = 0
    loss_i = 0 # Current Loss
    loss_log = [] # Loss history
    loss_avg = 0 # 100 iteration loss average

    # training loop
    for epoch in range(nb_epochs):
        for i in range(train_data.shape[0]-1):
            x = Variable(np.asarray(train_data[i], np.float32).reshape(1,1))
            t = Variable(np.asarray(train_data[i+1], np.float32).reshape(1,1))
            # Foward Pass
            x.to_gpu()
            t.to_gpu()
            loss_i = model(x,t)

            # Loss aggregation
            accumulated_loss += loss_i

            # Track the average loss every 100 iterations.
            loss_avg += accumulated_loss.data
            if (i % 100) == 0:
                loss_log.append(np.mean(loss_avg))
                loss_avg = 0

            #Backward pass and update
            if (i+1) % backprop_timer == 0:
                model.cleargrads()
                accumulated_loss.backward()
                accumulated_loss.unchain_backward()
                accumulated_loss.data.fill(0)
                opt.update()

        if epoch % 10 == 0:
            print('Epoch: ', epoch)

    return loss_log


# The evaluation loop
def evaluate(model,num, origin=0):
    evaluator = model.copy()

    # Don't store computation history
    evaluator.volatile = 'on'

    # Reset memory
    evaluator.mem = np.ones(
        (1,evaluator.h2h.W.data.shape[0]),
        dtype=np.float32)

    # Evaluate for a single time step to get first output
    p = evaluator(np.array([[origin]],np.float32))
    preds = []

    # Feed that into the next timestep and repeat for num iterations
    for i in range(num):
        p = evaluator(p)
        preds.append(p.data[0])

    return preds

gpu_id = 0
cuda.get_device(gpu_id).use()
data = np.arange(0,1000, 0.1, dtype=np.float32)

data[0:10]

train_data = np.sin(data)
# define model
model_0 = SimpleRNN0(1,3)
model_0C = L.Classifier(model_0,F.mean_squared_error)
model_0C.compute_accuracy = False
# load optimizer
opt = optimizers.Adam()
opt.setup(model_0C)
model_0C.to_gpu()
loss = train_SimpleRNN0(model_0C,50)
loss[-1]

