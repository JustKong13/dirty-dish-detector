import torch 
from torch import nn 
import torch.nn.functional as F
from model import model
from dataset_generation import train_loader, test_loader
from torch import optim
import pickle as pkl
import copy 


def val(model, val_data_loader, criterion):
    val_running_loss = 0
    accuracy = 0
    counter = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_data_loader, 0):
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(torch.long))
            val_running_loss += loss.item()
            output_decisions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(output_decisions == labels)
            counter += 1 
    accuracy = accuracy / counter

    return val_running_loss, accuracy

def train(model, data_loader, val_data_loader, criterion, optimizer, epochs):
    train_loss_arr = []
    val_loss_arr = []
    pred_arr = []
    running_loss = 0.0
    count = 0

    result_model = None

    for epoch in range(epochs):
        running_loss = 0.0
        num_correct = 0.0
        count = 0
        for batch_idx, (train_features, train_labels) in enumerate(data_loader):
            count += 1

            if count >= 15:
                break

            optimizer.zero_grad()
            preds = model(train_features) # saving all of our features
            train_loss = criterion(preds, train_labels.to(torch.long))

            train_loss.backward()

            running_loss += train_loss.item()

            optimizer.step()
            print('Current Loss at batch {}:'.format(batch_idx), str(running_loss))

        train_loss = running_loss / len(data_loader)
        train_loss_arr.append(train_loss)
        print("Current Loss at Epoch {}: ".format(epoch) + str(running_loss))

        val_loss, accuracy = val(model, val_data_loader, criterion)
        val_loss_arr.append(val_loss)
        pred_arr.append(accuracy)

        if val_loss < min(val_loss_arr): 
            result_model = copy.deepcopy(model)

    print('Training finished.')
    return train_loss_arr, val_loss_arr, pred_arr, result_model