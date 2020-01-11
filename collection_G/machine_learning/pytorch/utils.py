'''
Utilities
'''

import time
import copy

try:
    import torch
except:
    torch = None
    
from .history import History
from .._exception import *

class PyTorchTrainer:
    '''
    class for training models for pytorch
    '''
    def __init__(self,
        model,
        optimizer,
        criterion,
        scheduler=None,
        use_gpu=True
    ):
        '''
        Constructer

        arg
            model
                The model to train.
            optimizer
                The optimizer function
            criterion
                The loss function
            scheduler
                The scheduler
            use_gpu
                If True, tries to find visible GPU(s).
                It uses all visible GPUs if more than one.
                If False, uses CPU.
        '''

        if torch == None:
            raise_no_module_error('PyTorch')

        self.model, self.device = self._device_specify(model=model, use_gpu=use_gpu)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.history = History()

    def train(self,
        epochs,
        train_data,
        val_data,
        save=True,
        filename='./best_state_dict.pickle',
        verbose=True
    ):
        '''
        Training function

        Validation data must be prepaired.

        arg
            epochs
                The epochs to train the model.
                The training will not stop until the end.
            train_data
                The data for training.
                Must be a DataLoader object.
            val_data
                The data for validation
                Must be a DataLoader object
            save
                If True the best weights will be save to 'filename'.
            filename
                The filename to save the best weights
            verbose
                If True log for each epoch will be printed
        '''
        history = History()
        history.save_path = filename
        best_loss = 100.

        # training
        for epoch in range(1, epochs+1):
            print('EPOCH {} / {}'.format(epoch, epochs))
            epoch_start = time.time()

            # training phase
            phase = 'train'
            self.model.train(True)
            dataset = train_data
            loss        = 0
            correct     = 0
            # loop for batch
            for index, (data, target) in enumerate(dataset, 1):
                
                batch_loss, correct_count = self.batch_train(data=data, target=target)
                
                # summing up loss and correct prediction
                loss    += batch_loss.item()
                correct += correct_count
            # epoch loss and accuracy calculation
            self.history.history[phase]['loss'].append(loss / len(dataset))
            self.history.history[phase]['accuracy'].append(100 * correct / len(dataset.dataset))

            # validation phase
            phase = 'val'
            self.model.train(False)
            dataset = validation_data
            with torch.no_grad():    # for less memory usage
                loss    = 0
                correct = 0
                # loop for batch
                for index, (data, target) in enumerate(dataset, 1):
                    
                    batch_loss, correct_count = self.batch_validate(data, target)

                    loss    += batch_loss.item()
                    correct += correct_count

                # saving epoch loss and accuracy
                self.history.history[phase]['loss'].append(loss / len(dataset))
                self.history.history[phase]['accuracy'].append(100 * correct / len(dataset.dataset))
            
            # verbose
            if verbose:
                print('sec {:.2f}[s]'.format(time.time() - epoch_start),                  end='\t')
                print('Train loss : {:.5f}'.format(self.history.history['train']['loss'][-1]), end='\t')
                print( 'Train Acc : {:.5f}'.format(self.history.history['train']['accuracy'][-1]), end='\t')
                print(  'Val loss : {:.5f}'.format(self.history.history['val']['loss'][-1]),       end='\t')
                print(   'Val Acc : {:.5f}'.format(self.history.history['val']['accuracy'][-1]))

            # saving model
            # looking at validation loss
            if self.history.history['val']['loss'][-1] < best_loss:
                if save_model:
                    # deepcopy, for return
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    # output to binary file
                    torch.save(self.model.state_dict(), save_path)
                    print('Model saved to {}'.format(save_path))
                self.history.best_epoch = epoch + 1
                best_loss = self.history.history['val']['loss'][-1]

        # loading best weights to model
        self.model.load_state_dict(best_state_dict)
        
    def eval(self,
        test_data
    ):
        '''
        Evaluation function

        arg
            test_data
                The data for testing
        '''
        loss    = 0
        correct = 0
        
        self.model.eval()
        with torch.no_grad():
            for index, (data, target) in enumerate(test_data):

                batch_loss, correct_count = self.batch_validate(data, target)

                loss    += batch_loss.item()
                correct += correct_count
        print('Test Loss     : {:.5f}'.format(loss/len(test_dataloader.dataset)))
        print('Test Accuracy : {:.5f}'.format(100*correct/len(test_dataloader.dataset)))

    def batch_train(self,
        data,
        target,
    ):
        '''
        The training function for each batch.
        Over-ride this function to customize the training.

        arg
            data
                The data for one batch
            target
                The target of the data

        return
            batch_loss
                The loss calculated by the criterion
            correct_count
                The count of the correctly predicted samples
        '''
        data   = data.type(torch.FloatTensor).to(self.device)
        target = target.to(self.device)

        output = self.model(data)
        batch_loss = self.criterion(output, target)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        _, prediction = output.max(1)
        correct_count += (prediction == target).sum().item()

        return batch_loss, correct_count
    
    def batch_validate(self,
        data,
        target
    ):
        '''
        The validation function for each batch.
        Over-ride this function to customize the training.
        This function will also be used in the evaluation function.

        arg
            data
                The data for one batch
            target
                The target of the data

        return
            batch_loss
                The loss calculated by the criterion
            correct_count
                The count of the correctly predicted samples
        '''
        data   = data.type(torch.FloatTensor).to(self.device)
        target = target.to(self.device)

        output = self.model(data)
        batch_loss = self.criterion(output, target)

        _, prediction = output.max(1)
        correct_count = (prediction == target).sum().item()

        return batch_loss, correct_count

    def _device_specify(self, model, use_gpu=True):
        '''
        Function for specifying the device.
        
        arg
            model
                The model to train
            use_gpu
                If True, tries to find visible GPU(s).
                It uses all visible GPUs if more than one.
                If False, uses CPU.
        
        return
            model
                The model to train.
                If any GPU is visible, the model will be sent to the GPU.
            device
                The device that the model is on.
        '''
        if torch.cuda.is_available() and use_gpu:
            device = torch.device('cuda:0')
            if torch.cuda.device_count() > 0:
                print('Using {} GPUs.'.format(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model)
            model.to(device)
        else:
            device = torch.device('cpu')

        return model, device
