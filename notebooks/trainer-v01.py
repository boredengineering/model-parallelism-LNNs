import math
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

# ****** Step 2 / ToDo 1: Import the required library here ******
import deepspeed
# import uip

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers)


        def run_epoch(loader, is_train):
            model.train(is_train)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                x = x.to(self.device)
                y = y.to(self.device)
               
                # forward the model
                with torch.set_grad_enabled(is_train):
                    # ToDo Step 2: We will need to use the model_engine instead for the forward pass
                    logits, loss = model(x, y)
                    # logits, loss = model_engine(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    
                if is_train:

                    model.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
            
        for epoch in range(config.max_epochs):
            run_epoch(train_loader, is_train=True)


class DeepSpeedTrainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
            
    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model

        # ****** Step 2 / ToDo 2: Initialize deepspeed and obtain the model_engine and optimizer and train_loader
        #                         One of the things that you will need to initialize are the cmd_args 
        #                         Please pass them through the Trainer Config 
        trainset = self.train_dataset
        
        ## parameters = #FIXME
        # parameters = filter(lambda p: p.FIXME, model.FIXME())
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        
        model_engine, optimizer, train_loader, _ = deepspeed.initialize(
            args=config.cmd_args, 
            model=model, 
            model_parameters=parameters, 
            training_data=trainset)
        ## model_engine, optimizer, train_loader, _ = #FIXME        
        
        def run_epoch(loader, is_train):
            model.train(is_train)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # ****** Step 2 / ToDo 3: Change the below to place data on the local_rank device
                ## x = x.to(#FIXME)
                ## y = y.to(#FIXME)
                x = x.to(model_engine.local_rank)
                y = y.to(model_engine.local_rank)
                
                # forward the model
                with torch.set_grad_enabled(is_train):
                    # ****** Step 2 / ToDo 4: Use model_engine instead of model in the forward pass
                    ## logits, loss = #FIXME  
                    logits, loss = model_engine(x, y)  
                    ## loss = loss.mean() # collapse all losses if they are scattered on multiple gpus   
                    losses.append(loss.item())
                    
                if is_train:
                    # backprop and update the parameters
                    
                    #  ****** Step 2 / ToDo 5: Use model_engine instead of model for the backward pass
                    model_engine.backward(loss)

                    
                    #  ****** Step 2 / ToDo 6: Use model_engine instead of model for the optimizer step
                    model_engine.step()
                    
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
            
        for epoch in range(config.max_epochs):
            run_epoch(train_loader, is_train=True)
