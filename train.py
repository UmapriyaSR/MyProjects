import torchvision.datasets as datasets #for dataset built in functions
import matplotlib.pyplot as plt #for visualization
import utils

from utils import *
#load data and applying transform for uniform images
image_dataset = datasets.ImageFolder('./dataset', transform=data_transform) 

#define train and validation size
#split train/val : 80/20
train_size= int(0.8*len(image_dataset))
val_size = len(image_dataset) - train_size

training_set, validation_set = torch.utils.data.random_split(image_dataset, [train_size, val_size])
#training_set, validation_set = torch.utils.data.random_split(image_dataset, [1928, 847])

print('Train set:', len(training_set))
print('Validation set:', len(validation_set))

#Load dataset
batch_size = 32
training_load = torch.utils.data.DataLoader(image_dataset=training_set, batch_size=batch_size, shuffle=True)
validation_load = torch.utils.data.DataLoader(image_dataset=validation_set, batch_size=batch_size, shuffle=False)

#Show img after load
def imageshow(image):
     image = image/2 + 0.5 #unnormalize 
     np_image = image.numpy()
     plt.figure(figsize=(20, 20))
     plt.imshow(np.transpose(np_image, (1, 2, 0)))
     plt.show()

#to test the validation set
data_iteration = iter(validation_load)
image, labels = data_iteration.next()
 #imgshow(torchvision.utils.make_grid(img))

model = CNN() #calling the CNN

#create empty loss and accuracy lists
train_lossfunc = []
val_lossfunc = []
train_accuracy = []
val_accuracy = []

def Training_Model(model, epochs, parameters):
    #Using CrossEntropyLoss, SGD optimiser
    loss_func = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(parameters, lr=0.01) 

    model = model.cuda() #enable GPU operations
    
    for epoch in range(epochs): 
        start = time.time() 
        correct = 0
        iterations = 0
        iteration_loss = 0.0
        
        model.train() #Set mode Train                  
        #get the imputs
        for i, (inputs, labels) in enumerate(training_load, 0):
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            #Convert to Cuda() to use GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            # # zero the parameter gradients
            optimizer.zero_grad()  
            
       # predict classes using images from the training set
       #forward propogation
            outputs = model(inputs)
            
            #Calculating loss based on model output and real labels
            loss = loss_func(outputs, labels)  
            iteration_loss += loss.item()
            
            #Backpropagation
            loss.backward()              
            optimizer.step()             
            
            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs, 1)
            # adjust parameters based on the calculated gradients
            correct += (predicted == labels).sum()
            iterations += 1
    

        train_lossfunc.append(iteration_loss/iterations)
        train_accuracy.append((100 * correct / len(training_set)))
   

        #Evaluation on validation set
        loss = 0.0
        correct = 0
        iterations = 0

        model.eval() #Set mode evaluation

        #No_grad on Val_set
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(validation_load, 0):
                
                inputs = Variable(inputs)
                labels = Variable(labels)
                
                #To Cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                #Forward and Caculating loss
                outputs = model(inputs)     
                loss = loss_func(outputs, labels) 
                loss += loss.item()

                # Record the correct predictions for val data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1

            validation_lossfunc.append(loss/iterations)
            validation_accuracy.append((100 * correct / len(validation_set)))

        stop = time.time()
        
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}, Time: {}s'
            .format(epoch+1, epochs, train_lossfunc[-1], train_accuracy[-1], val_lossfunc[-1], val_accuracy[-1],stop-start))

epochs = 32
Training_Model(model=model, epochs=epochs, parameters=model.parameters())

#Save model
torch.save(model.state_dict(), 'weights/Face-Mask-Model.pt')

#Show chart acc and save Acc_chart
#plt.plot(train_acc, label='Train_Accuracy')
#plt.plot(val_acc, label='Val_Accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.axis('equal')
#plt.legend(loc=7)
#plt.savefig('Acc_chart.png')
#plt.show()
