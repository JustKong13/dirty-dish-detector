from model import model
from train import * 
from dataset_generation import * 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-2)
epochs = 4
train_loss, val_loss, pred, result_model = train(model, 
                                                 train_loader, 
                                                 test_loader, 
                                                 criterion, 
                                                 optimizer, 
                                                 epochs)

print('train loss', train_loss)
print('val_loss', val_loss)
print('pred', pred)


pkl.dump(result_model, open('tuned_model_resnet.pkl', 'wb'))