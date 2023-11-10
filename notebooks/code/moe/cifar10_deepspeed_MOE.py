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
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

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

    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=2,
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

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    
    
    parser.add_argument('--num-experts-per-layer',
                        type=int,
                        default=1,
                        help='number of experts per layers, MoE related.')
    
    parser.add_argument('--mlp-type',
                        type=str,
                        default='standard',
                        help='Only applicable when num-experts > 1, accepts [standard, residual]')
    
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    
    parser.add_argument('--min-capacity',
                        default=0,
                        type=int,
                        help='(moe) minimum capacity of an expert regardless of the capacity_factor')
    
    parser.add_argument('--noisy-gate-policy',
                        default=None,
                        type=str,
                        help='(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter')
    
    parser.add_argument('--moe-param-group',
                        default=False,
                        action='store_true',
                        help='(moe) create separate moe param groups, required when using ZeRO w. MoE')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

deepspeed.init_distributed() 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()
    
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    torch.distributed.barrier()
    
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



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if args.moe: # changed here
            fc3 = nn.Linear(84, 84)
            self.experts = deepspeed.moe.layer.MoE(hidden_size=84,expert=fc3, num_experts=args.num_experts_per_layer, ep_size=args.ep_world_size, use_residual=args.mlp_type == 'residual',  k=args.top_k, min_capacity=args.min_capacity, noisy_gate_policy=args.noisy_gate_policy)
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if args.moe: # changed here
            x, _, _ = self.experts(x) 
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x



net = Net()


def create_moe_param_groups(model):


   

    return 


if args.moe_param_group:
    params = {'params': [p for p in net.parameters()],
        'name': 'parameters'
        }
    parameters = split_params_into_different_moe_groups_for_optimizer(params)
else:
    parameters = filter(lambda p: p.requires_grad, net.parameters())


model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset)

fp16 = model_engine.fp16_enabled()
device = model_engine.local_rank


criterion = nn.CrossEntropyLoss()
writer = SummaryWriter('/dli/megatron/tensorboard/cifar10_DS' + str(args.profile_name))
with ExitStack() as stack:
    if args.profile_execution and torch.distributed.get_rank()==0:
        prof = stack.enter_context(torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('/dli/megatron/tensorboard/cifar10_' + str(args.profile_name)+ str(torch.distributed.get_rank())),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        with_flops=True
                    ))

    for epoch in range(args.epochs): 
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
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


dataiter = iter(testloader)
images, labels = dataiter.next()



class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(model_engine.local_rank)).sum().item()
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print('Accuracy of the network on the 10000 test images: %2f %%' %
      (100 * correct / total))
for i in range(10):
    print('Accuracy of %5s : %2f %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))
    
print('Evaluation Done')
