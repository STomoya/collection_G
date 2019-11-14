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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    plt = None

from ._exception import *

# TODO: Make a function for 2 input 1 output model for research on multimodal representation
# NOTE: separating cuda settings from functions might make saving and loading parameters easier

def torch_train_flow(
    model,
    optimizer,
    criterion,
    epochs,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    save_model_path='./model_param',
    save_img_path='./loss_acc_curve.png',
    verbose=True
):
    """
    Runs the whole training sequence for PyTorch models

    flow
        train
            ->
            plot loss and accuracy curve
                ->
                load best parameters to model
                    ->
                    evaluate

    # TODO: TEST THIS FUNCTION
    
    argument
        model
            The model to train
        optimizer
            The optimizer function to train the model
        criterion
            The loss function to train the model
        epochs
            The number of epochs to run
        *_dataloader
            The data loaders, corresponding to the name
        save_model_path
            The path to save the model
        save_img_path
            The path to save the figure
            the filename must have '.png'
        verbose
            If True The loss and accuracy of each epoch
            will be printed to sys.stdout.
    """

    # train model
    history = torch_fit(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        validation_dataloader=validation_dataloader,
        save_model=True,
        save_path=save_model_path,
        verbose=verbose
    )

    # print best epoch
    print('Best Epoch : {}'.format(history.best_epoch))

    # plot history
    plot_torch_history(
        history=history,
        show=False,
        save=True,
        filename=save_img_path
    )

    # load the best parameters to the model
    state_dict = torch.load(history.save_path)
    # NOTE:
    # If the saved model was on GPU the keys contain 'module.',
    # which should be erased to be used on to models on CPU
    if torch.cuda.is_available():
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name = key[7:]
            new_state_dict[name] = value
        state_dict = new_state_dict
    model.load_state_dict(state_dict)

    torch_eval(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion
    )


def torch_fit(
    model,
    train_dataloader,
    optimizer,
    criterion,
    epochs,
    validation_dataloader=None,
    save_model=True,
    save_path='./model_param',
    verbose=True,
):
    """
    Fit function for pytorch

    You can train pytorch models with this function.

    This function only allows a single input and a single output.
    If you are using a multi-input or multi-output model,
    you will have to write your oun training function.

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
        save_model
            If True the model parameters will be saved,
            using torch.save(model.state_dict(), save_path)
        save_path
            The path to save the best parameters of the model
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
    history.save_path = save_path
    if not validation_dataloader == None:
        phases = ['train', 'val']
    else:
        phases = ['train']
    best_loss = 100.

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
                data   = data.type(torch.FloatTensor).to(device)
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
                _, prediction = output.max(1)
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
        if history.history['val']['loss'][-1] < best_loss:
            if save_model:
                torch.save(model.state_dict(), save_path)
                print('Model saved to {}'.format(save_path))
            history.best_epoch = epoch + 1
            best_loss = history.history['val']['loss'][-1]

    return history

def torch_eval(
    model,
    test_dataloader,
    criterion
):
    """
    Evaluation function for PyTorch models

    Only for one input, one output model

    argument
        model
            Model to evaluate
        test_dataloader
            torch.utils.data.DataLoader object for testing
        criterion
            Loss function
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

    correct = 0
    loss    = 0
    model.train(False)
    with torch.no_grad():
        for (data, target) in test_dataloader:
            data   = data.type(torch.FloatTensor).to(device)
            target = target.to(device)

            output = model(data)
            batch_loss = criterion(output, target)
            loss += batch_loss.item()
            _, predicted = output.max(1)
            correct += (predicted == target).sum().item()
    print('Test Loss     : {:.5f}'.format(loss/len(test_dataloader.dataset)))
    print('Test Accuracy : {:.5f}'.format(100*correct/len(test_dataloader.dataset)))

def plot_torch_history(
    history,
    show=True,
    save=True,
    filename='./torch_history.png'
):
    """
    Plotter for torch history

    Cannot plot history function
    that dose not contain information about validation.

    # TODO: erase show option. we will never show figures anymore.

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
    if show:
        pass
    
    plt.close()


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
        self.save_path = None
        self.best_epoch = None


