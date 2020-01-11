'''
history class for logging training
'''

try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
except:
    plt = None

from .._exception import *

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

    def plot(self, filename='training_curve.png'):

        if plt == None:
            raise_no_module_error('matplotlib')

        train_loss = self.history['train']['loss']
        val_loss   = self.history['val']['loss']
        train_acc  = self.history['train']['accuracy']
        val_acc    = self.history['val']['accuracy']
        
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

        plt.savefig(filename)
        
        plt.close()
        

