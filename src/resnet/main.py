from model import model
from train import * 
from dataset_generation import * 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
epochs = 8
train_loss, val_loss, train_pred, pred, result_model = train(model, 
                                                 train_loader, 
                                                 test_loader, 
                                                 criterion, 
                                                 optimizer,
                                                 scheduler, 
                                                 epochs)

print('train loss', train_loss)
print('val_loss', val_loss)
print("train accuracy", train_pred)
print('validation accuracy', pred)


# pkl.dump(result_model, open('unbias_tuned_model_resnet.pkl', 'wb'))
torch.save(result_model.state_dict(), './models/v8_unbias_tuned_model_resnet.pt')