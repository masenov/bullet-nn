from chainer import Chain
from chainer import Variable
from chainer import datasets
from chainer import optimizers
from chainer import cuda
from chainer import iterators
from chainer import training, report
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import numpy as np



model_0 = Chain(conv0 = L.Convolution2D(in_channels=1, out_channels=16, ksize=3, stride=1, pad=0, wscale=1, bias=0),

                            conv1 = L.Convolution2D(16,32,3),
                            conv2 = L.Convolution2D(32, 32, 3, pad=1),
                            conv3 = L.Convolution2D(32, 32, 3, pad=1),
                            conv4 = L.Convolution2D(32, 32, 3),
                            FC1 = L.Linear(512,100),
                            FC2 = L.Linear(100,10))

##### We could have just done this instead too
#conv0 = L.Convolution2D(in_channels=1, out_channels=16, ksize=3, stride=1, pad=0, wscale=1, bias=0)
#conv1 = L.Convolution2D(16,32,3),
#conv2 = L.Convolution2D(32, 32, 3, pad=1),
#conv3 = L.Convolution2D(32, 32, 3, pad=1),
#conv4 = L.Convolution2D(32, 32, 3),
#FC1 = L.Linear(512,100),
#FC2 = L.Linear(100,10)



def foward_0(x ,t=None, predict = False):
    l0 = model_0.conv0(x)
    l0 = F.relu(l0)

    l1 = model_0.conv1(l0)
    l1 = F.relu(l1)

    l2 = model_0.conv2(l1)
    l2 = F.max_pooling_2d(F.relu(l2),2)

    l3 = model_0.conv3(l2)
    l3 = F.max_pooling_2d(F.relu(l3),2)

    l4 = model_0.conv4(l3)
    l4 = F.relu(l4)

    l5 = model_0.FC1(l4)
    l5 = F.relu(l5)

    l6 = model_0.FC2(l5)
    l6 = F.relu(l6)

    y = l6

    if not predict:
        return F.softmax_cross_entropy(y,t)

    return F.softmax(y).data.argmax(1)

def mnist_train_0(data, test, nb_epochs = 10):

    for epoch in range(nb_epochs):
        print ("Current epoch: %d"%(epoch + 1))

        # clear gradient array
        model_0.cleargrads()

        # import subset of the data into numpy array with proper types
        subset =  [i for i in range(500)]
        x = np.array(data[subset][0]).astype(np.float32)
        y = np.array(data[subset][1]).astype(np.int32)

        # reshape it for chainer and cast to chainer variable
        x = x.reshape(len(subset),1,28,28)
        x = Variable(x)


        # evaluate data on model and backpropagate
        loss = foward_0(x, y)
        loss.backward()

        # update model parameters
        optimizer.update()



        ### evaluate on testing set ###
        # import data into numpy array with proper types
        subset = [i for i in range(100)]
        x = np.array(test[subset][0]).astype(np.float32)
        y = np.array(test[subset][1]).astype(np.int32)

        # reshape and cast to chainer variable
        x = Variable(x.reshape(len(subset),1,28,28))

        # evaluate test data using the current network parameters
        pred = foward_0(x,None,predict=True)

        # calculate accuracy
        acc = (pred == y).mean()
        print ( "Accuracy : {} \nError Rate: {}".format(acc * 100, (1-acc) * 100))

def mnist_train_0_batch(data, test, batch_size = 64, nb_epochs = 10):

    for epoch in range(nb_epochs):
        print ("Current epoch: %d"%(epoch + 1))

        ## shuffle the dataset
        nb_data = len(data) - (len(data) % batch_size)
        shuffler = np.random.permutation(nb_data)

        for i in range(0, nb_data, batch_size):

            # clear or zero-out gradients
            model_0.cleargrads()

            # import subset of the data into numpy array with proper types
            x = np.array(data[shuffler[i : i + batch_size]][0]).astype(np.float32)
            y = np.array(data[shuffler[i : i + batch_size]][1]).astype(np.int32)

            # reshape for channel depth dimension and cast to chainer variable
            x = x.reshape(batch_size,1,28,28)
            x = Variable(x)


            # evaluate data on model and backpropagate
            loss = foward_0(x, y)
            loss.backward()

            # update model parameters
            optimizer.update()



        ### evaluate on entire testing set ###
        print("Validation Set")
        # import data into numpy array with proper types
        x = np.array(test[:,][0]).astype(np.float32)
        y = np.array(test[:,][1]).astype(np.int32)

        # reshape and cast to chainer variable
        x = Variable(x.reshape(len(test),1,28,28))

        pred = foward_0(x,None,predict=True)

        acc = (pred == y).mean()
        print ( "Accuracy : {} \nError Rate: {}".format(acc * 100, (1-acc) * 100))


def mnist_train_0_gpu(data, test, batch_size=128, nb_epochs = 10):

    for epoch in range(nb_epochs):

        print ("Current epoch: %d"%(epoch + 1))

        ## shuffle the dataset
        nb_data = len(data) - (len(data) % batch_size)
        shuffler = np.random.permutation(nb_data)

        for i in range(0, nb_data, batch_size):

            # clear previous gradients
            model_0.cleargrads()

            # import data, normalise and reshape it for batch processing
            x = np.array(data[shuffler[i : i + batch_size]][0]).astype(np.float32).reshape(batch_size,1,28,28)
            y = np.array(data[shuffler[i : i + batch_size]][1]).astype(np.int32).reshape(batch_size)  

            # send data arrays to gpu
            x = Variable(cuda.to_gpu(x))
            y = Variable(cuda.to_gpu(y))

            # evaluate data on model and backpropagate
            loss = foward_0(x,y)
            loss.backward()

            # update model parameters
            optimizer.update()

        ### evaluate on testing set ##
        mnist_evaluation(test)

def mnist_train_0_gpu(data, test, batch_size=128, nb_epochs = 10):
    for epoch in range(nb_epochs):

        print ("Current epoch: %d"%(epoch + 1))

        ## shuffle the dataset
        nb_data = len(data) - (len(data) % batch_size)
        shuffler = np.random.permutation(nb_data)

        for i in range(0, nb_data, batch_size):

            # import data, normalise and reshape it for batch processing
            x = np.array(data[shuffler[i : i + batch_size]][0]).astype(np.float32)
            y = np.array(data[shuffler[i : i + batch_size]][1]).astype(np.int32) 

            # send data arrays to gpu
            x = Variable(cuda.to_gpu(x))
            y = Variable(cuda.to_gpu(y))

            # automatically calls cleargrads -> forward pass --> backward pass --> parameter update
            optimizer.update(foward_0,x,y)

        ### evaluate on testing set ##
        mnist_evaluation(test)


### evaluate on entire testing set ###
def mnist_evaluation(test):
        # import data into numpy array with proper types
        x = np.array(test[:,][0]).astype(np.float32)

        x = Variable(cuda.to_gpu(x))
        y = cuda.to_gpu(test[:,][1])

        pred = foward_0(x,None,predict=True)

        acc = (pred == y).mean()
        print ( "Accuracy : {} \nError Rate: {}".format(acc * 100, (1-acc) * 100))


def model_statistics(model):

    # check if model is on the gpu or cpu
    on_cpu = model._cpu
    if not on_cpu: model.to_cpu()

    # iterate over layers and model parameters
    count, size = 0, 0

    for layer in model.children():
        if layer.W is not None:
            count += layer.W.size
            size += layer.W.data.nbytes

        if layer.b is not None:
            count += layer.b.size
            size += layer.b.data.nbytes


    # Return model to gpu, if it was there originally
    if on_cpu != model._cpu: model.to_gpu()
    return count, round(size/1048576.0,4)


class SimpleCNN0(Chain):

    def __init__(self):

        super().__init__(
            conv0 = L.Convolution2D(1, 16, 3),
            conv1 = L.Convolution2D(16,32,3),
            conv2 = L.Convolution2D(32, 32, 3, pad=1),
            conv3 = L.Convolution2D(32, 32, 3, pad=1),
            conv4 = L.Convolution2D(32, 32, 3),
            FC1 = L.Linear(512,100),
            FC2 = L.Linear(100,10)
        )

    def foward(self,x):

        l0 = self.conv0(x)
        l0 = F.relu(l0)

        l1 = self.conv1(l0)
        l1 = F.relu(l1)

        l2 = self.conv2(l1)
        l2 = F.max_pooling_2d(F.relu(l2),2)

        l3 = self.conv3(l2)
        l3 = F.max_pooling_2d(F.relu(l3),2)

        l4 = self.conv4(l3)
        l4 = F.relu(l4)

        l5 = self.FC1(l4)
        l5 = F.relu(l5)

        l6 = self.FC2(l5)
        l6 = F.relu(l6)

        return l6

    def __call__(self,x,t):

        y = self.foward(x)

        loss = F.softmax_cross_entropy(y,t)
        accuracy = F.accuracy(y,t)

        ## Model stats
        nb_params = model_statistics(self)

        ## Report back the loss and accuracy at every iteration
        report({'loss' : loss, 'accuracy': accuracy, 'nb_params': nb_params[0]}, self)

        return loss

train, test = datasets.get_mnist(ndim=2)
import pdb; pdb.set_trace()
# Setup Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model_0)

# mnist_train_0(train,test,5)
# mnist_train_0_batch(train,test, nb_epochs = 1)  ## Takes a very long time


train, test = datasets.get_mnist(ndim=3) # Return the channel dimension too
# Send the model to the gpu
gpu_id = 0
cuda.get_device(gpu_id).use()
model_0.to_gpu()
# mnist_train_0_gpu(train,test, nb_epochs=5)


stats = model_statistics(model_0)
print("Number of parameters: ",stats[0], "Memory usage in MiB:", stats[1])

model_1 = SimpleCNN0()
model_1.name =  "SimpleCNN0"

## Setup optimizer
optim = optimizers.Adam()
optim.setup(model_1)
optim.use_cleargrads()
## Send to GPU
model_1.to_gpu()


# batch_size = 128
# train_iter = iterators.SerialIterator(train,batch_size)
# test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

# ## updater takes the data iterator, optimiser with attached model and computation device id (the gpu)
# updater = training.StandardUpdater(train_iter, optim, device=gpu_id)
# ## trainer runs through several epochs/iterations of training
# trainer = training.Trainer(updater,(5,"epoch"))
# ## trainer extension that evalutes on the test set
# trainer.extend(extensions.Evaluator(test_iter, model_1,device=gpu_id))

# trainer.run()




batch_size = 128
train_iter = iterators.SerialIterator(train,batch_size)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optim, device=gpu_id)
trainer = training.Trainer(updater,(2,"epoch"))
trainer.extend(extensions.Evaluator(test_iter, model_1,device=gpu_id))
trainer.extend(extensions.LogReport()) # passed directly to trainer

## Print the report
trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time', 'iteration' , 'main/accuracy', 'validation/main/accuracy',  'main/nb_params']))

trainer.extend(extensions.ProgressBar(update_interval=300))
## Run trainer
trainer.run()

