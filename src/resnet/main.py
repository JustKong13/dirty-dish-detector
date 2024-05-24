from model import model
from train import * 
from dataset_generation import * 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
epochs = 5
train_loss, val_loss, train_pred, pred, result_model = train(model, 
                                                 train_loader, 
                                                 test_loader, 
                                                 criterion, 
                                                 optimizer, 
                                                 epochs)

print('train loss', train_loss)
print('val_loss', val_loss)
print("train accuracy", train_pred)
print('validation accuracy', pred)


# pkl.dump(result_model, open('unbias_tuned_model_resnet.pkl', 'wb'))
torch.save(result_model.state_dict(), './models/v3_unbias_tuned_model_resnet.pt')