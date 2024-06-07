import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import torch.profiler
from contextlib import ExitStack

def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help="output logging information at a given interval")
    
    parser.add_argument('--profile-execution', type=bool, default=False,
                       help='Use pytorch profiler during execution ')
    
    parser.add_argument('--profile-name', default=False,
                       help=' Profile folder name ') 

    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



args = add_argument()

net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
parameters = filter(lambda p: p.requires_grad, net.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset)

fp16 = model_engine.fp16_enabled()
device = model_engine.local_rank

criterion = nn.CrossEntropyLoss()
writer = SummaryWriter('/dli/nemo/resnet152_DS_' + str(args.profile_name))
with ExitStack() as stack:
    if args.profile_execution and torch.distributed.get_rank()==0:
        prof = stack.enter_context(torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('/dli/nemo/resnet152_' + str(args.profile_name)+'_gpu' +str(torch.distributed.get_rank())),
                    profile_memory=True
                ))

    for epoch in range(args.epochs): 
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            if fp16:
                inputs = inputs.half()        
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()


            # print the loss and accuracy metrics very log_interval mini-batches
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if i % args.log_interval == (args.log_interval - 1):  
                print('[epoch %d, iterations %5d] loss: %.3f accuracy: %2f %%' %  (epoch , i + 1, running_loss / args.log_interval, 100.*correct/total))
                writer.add_scalar("Training Cross Entropy Loss", running_loss / args.log_interval, i + 1)
                writer.add_scalar("Training Accuracy", 100.*correct/total, i + 1)
                running_loss = 0.0
            if args.profile_execution and torch.distributed.get_rank()==0:
                prof.step()

print('Training Done')
writer.add_graph(model_engine, inputs)
writer.flush()
writer.close()
