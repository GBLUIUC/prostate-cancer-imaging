import matplotlib.pyplot as plt
import numpy as np

def plot(optim, loss, train_loss, val_loss, train_acc, val_acc):
    fig = plt.figure()
    fig.set_size_inches(15, 5)
    ax1 = fig.add_subplot(121)
    ax1.title.set_text(optim + ' - ' + loss + ' Loss')
    ax1.plot(train_loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.title.set_text(optim + ' - ' + loss + ' Accuracy')
    ax2.plot(train_acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.legend()

    plt.show()