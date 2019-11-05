"""
Functions around machine learning

for pytorch
"""

import time

try:
    import torch
except:
    torch = None
try:
    import matplotlib.pyplot as plt
except:
    plt = None

from ._exception import *

# TODO:Make a function for 2 input 1 output model for research on multimodal representation

def torch_fit(
    model,
    train_dataloader,
    optimizer,
    criterion,
    epochs,
    validation_dataloader=None,
    verbose=True,
):
    """
    Fit function for pytorch

    You can train pytorch models with this function.

    This function only allows a single input and a single output.
    If you are using a multi-input or multi-output model,
    you will have to write your oun training function.

    TODO:TEST THIS FUNCTION
    TODO:Make multi input output possible
    TODO:add more frequent verbose

    argument
        model
            model to train
        train_dataloader
            torch.utils.data.DataLoader object for training
            The handled dataset must "return data, target"
        optimizer
            optimizer function
        criterion
            loss function
        epochs
            epochs for training
        validation_dataloader
            torch.utils.data.DataLoader object for validation
            The handled dataset must "return data, target"
        verbose
            If True, the log for each epoch will be outputed
    
    return
        model
            The trained model
            Must be catched or you will not have the trained model
        history
            The History class object
            Contains the loss and accuracy for train and validation
            on each epoch.
    """
    # pytorch must be installed
    if not torch:
        raise_no_module_error('torch')

    # initualizations
    # for gpu usage
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)
    history = History()
    if not validation_dataloader == None:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # training
    for epoch in range(1, epochs+1):
        print('EPOCH {} / {}'.format(epoch, epochs))
        epoch_start = time.time()
        for phase in phases:
            if phase == 'train':
                model.train(True)
                dataset = train_dataloader
            else:
                model.train(False)
                dataset = validation_dataloader
            loss        = 0
            correct     = 0
            batch_count = 0
            # loop for batch
            for index, (data, target) in enumerate(dataset, 1):
                # to device
                # gpu or cpu, depends on your env
                data   = data.to(device)
                target = target.to(device)

                # batch loss calculation
                output = model(data)
                batch_loss = criterion(output, target)

                # optimization
                # only on training
                if phase == 'train':
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                # summing up loss and correct prediction
                loss        += batch_loss.item()
                _, prediction = torch.max(output.data, 1)
                correct     += (prediction == target).sum().item()
                batch_count += 1
            # epoch loss and accuracy calculation
            history.history[phase]['loss'].append(loss / batch_count)
            history.history[phase]['accuracy'].append(100 * correct / len(dataset.dataset))
        
        # verbose
        if verbose:
            print('sec {:.2f}[s]'.format(time.time() - epoch_start),                  end='\t')
            print('Train loss : {:.5f}'.format(history.history['train']['loss'][-1]), end='\t')
            if not validation_dataloader == None:
                print( 'Train Acc : {:.5f}'.format(history.history['train']['accuracy'][-1]), end='\t')
                print(  'Val loss : {:.5f}'.format(history.history['val']['loss'][-1]),       end='\t')
                print(   'Val Acc : {:.5f}'.format(history.history['val']['accuracy'][-1]))
            else:
                print('Train Acc : {:.5f}'.format(history.history['train']['accuracy'][-1]))
    
    return model, history

def torch_eval(
    model,
    test_dataloader,
    criterion
):
    """
    Evaluation function for PyTorch models

    Only for one input, one output model

    TODO:finish making this function

    argument
        model
            Model to evaluate
        test_dataloader
            torch.utils.data.DataLoader object for testing
        criterion
            Loss function
    """


def plot_torch_history(
    history,
    save=True,
    filename='./torch_history.py'
):
    """
    Plotter for torch history

    Cannot plot history function
    that dose not contain information about validation.

    TODO:Make it possible to plot without having validation data

    argument
        history
            A History object.
            History object is returned from torch_fit function
        save
            If True, a png file will be saved
        filename
            The file name of the saving image
    """
    if not plt:
        raise_no_module_error('matplotlib')

    train_loss = history.history['train']['loss']
    val_loss = history.history['val']['loss']
    train_acc = history.history['train']['accuracy']
    val_acc = history.history['val']['accuracy']

    if len(val_acc) == 0:
        raise Exception('No validation')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.tight_layout()

    if save:
        plt.savefig(filename)
    plt.show()


class History:
    def __init__(self):
        self.history = {
            'train' : {
                'loss'     : [],
                'accuracy' : []
            },
            'val' : {
                'loss'     : [],
                'accuracy' : []
            }
        }
