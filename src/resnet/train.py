import torch 
from torch import nn 
import torch.nn.functional as F
from model import model
from dataset_generation import train_loader, test_loader
from torch import optim
import pickle as pkl
import copy 

def get_train_metrics(model, train_data_loader, criterion):
    model.eval()
    val_running_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in train_data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(torch.long))
            val_running_loss += loss.item()
            predictions = outputs.argmax(dim=1, keepdim=True)
            total_correct += predictions.eq(labels.view_as(predictions)).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    return val_running_loss / len(train_data_loader), accuracy

def val(model, val_data_loader, criterion):
    model.eval()
    val_running_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(torch.long))
            val_running_loss += loss.item()
            predictions = outputs.argmax(dim=1, keepdim=True)
            total_correct += predictions.eq(labels.view_as(predictions)).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    return val_running_loss / len(val_data_loader), accuracy

def train(model, data_loader, val_data_loader, criterion, optimizer, scheduler, epochs):
    train_loss_arr = []
    val_loss_arr = []
    train_pred_arr = []
    pred_arr = []
    running_loss = 0.0
    count = 0

    result_model = None
    min_val_loss = float('inf')

    for epoch in range(epochs):
        running_loss = 0.0
        num_correct = 0.0
        count = 0
        for batch_idx, (train_features, train_labels) in enumerate(data_loader):
            count += 1

            # if count >= 15:
            #     break
            
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

        _, accuracy = get_train_metrics(model, data_loader, criterion)
        train_pred_arr.append(accuracy)

        val_loss, accuracy = val(model, val_data_loader, criterion)
        val_loss_arr.append(val_loss)
        pred_arr.append(accuracy)

        if val_loss < min_val_loss: 
            min_val_loss = val_loss
            result_model = copy.deepcopy(model)
        
        scheduler.step()

    print('Training finished.')
    return train_loss_arr, val_loss_arr, train_pred_arr, pred_arr, result_model