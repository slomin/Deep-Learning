import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def append_global_history(global_history, history):
    global_history.params['epochs'] += history.params['epochs']
    number_of_epochs = global_history.params['epochs']
    global_history.epoch = list(range(0, number_of_epochs))
    global_history.history['val_loss'].extend(history.history['val_loss'])
    global_history.history['val_acc'].extend(history.history['val_acc'])
    global_history.history['loss'].extend(history.history['loss'])
    global_history.history['acc'].extend(history.history['acc'])
    return global_history

def display_nine_examples(x, y):
    plt.rcParams['figure.figsize'] = (6, 6)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(x[i], cmap='gray', interpolation='none')
        label = _load_label_names()[y[i][0]]
        plt.title("Label: {}".format(label))

def display_nine_random_predictions(predicted_classes, correct_or_incorrect, x_test, y_val_test):
    correct_or_incorrect = random.sample(list(correct_or_incorrect), 9) # choose 9 random indecies
    plt.figure()
    for i, index in enumerate(correct_or_incorrect):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        image = x_test[index].reshape(32, 32, 3)
        plt.imshow(image.astype('uint8'))
        predicted_label = _load_label_names()[predicted_classes[index]]
        actual_label = _load_label_names()[y_val_test[index][0]]
        plt.title("Pred: {}".format(predicted_label))
        plt.xlabel("Real: {}".format(actual_label))

def display_data_augmentation(datagen, x, y):
    datagen.fit(x[:9])
    for x_batch, y_batch in datagen.flow(x, y, batch_size=9):
        plt.figure()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            image = x_batch[i].reshape(32, 32, 3)
            plt.imshow(image.astype('uint8'))
            label = _load_label_names()[y_batch[i][0]]
            plt.title("Label: {}".format(label))
        plt.show()
        break


def display_accuracy_and_loss(history):
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def display_incorect_with_probabilities(model, incorrect, x_test, y_val_test, how_many_to_show):
    for x in range(how_many_to_show):
        i = random.sample(list(incorrect), 1)
        image = x_test[i].reshape(32, 32, 3)
        f = plt.figure()
        f.set_figwidth(7)
        plt.subplot(1, 2, 1)
        label = _load_label_names()[y_val_test[i][0][0]]
        plt.title('Label: {}'.format(label))
        plt.imshow(image.astype('uint8'))
        plt.axis('off')
        probs = model.predict_proba(image.reshape(1, 32, 32, 3), batch_size=1, verbose=0)
        plt.subplot(1, 2, 2)
        plt.title('Classification probabilities:')
        plt.bar(np.arange(10), probs.reshape(10), align='center')
        plt.xticks(np.arange(10), _load_label_names(), rotation = (45), fontsize = 12, va='bottom', ha='left')
