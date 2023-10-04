import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F

import time

import matplotlib.pyplot as plt
import numpy as np

from BaseNetwork import BaseNetwork
from Block1 import Block1
from Block2 import Block2

import argparse



def main():
    parser = argparse.ArgumentParser(description="A simple script with options")
    parser.add_argument("--model", help="Specify a model name (e.g., --model base)", required=False)
    parser.add_argument("--freeze", help="Specify whether to freeze Block1 weights (e.g., --freeze 1)", required=False)
    
    args = parser.parse_args()
    # Define data transformations to augment the dataset during training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Define data transformations for testing (no data augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the CIFAR-10 dataset and apply the transformations
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Define batch size for the data loaders
    batch_size = 128

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    #With these data loaders, you can now use the previous code to train and evaluate the model on the CIFAR-10 dataset. The training data will be augmented with random horizontal flips and random crops during each epoch, while the test data will be used for evaluation without augmentation. The data loaders will automatically handle the loading, batching, and shuffling of the data.


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaseNetwork().to(device)
    

    if args.model == "block1":
        model = Block1().to(device)
    elif args.model == "block2":
        freeze = 0
        if args.freeze == "1":
            freeze = 1
        model1 = Block1().to(device)
        model1.load_state_dict(torch.load("block1_trained_model.pth"))
        model = Block2(model1,freeze).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    loss_history = []
    num_epochs = 20
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        loss_history.append(loss.data)

    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    #Saving the trained model
    torch.save(model.state_dict(), args.model+"_trained_model.pth")
    # Plot the training loss over epochs
    loss_history_n = []
    for i in loss_history:
        loss_history_n.append(i.detach().cpu().numpy())
    # Plot the training loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history_n, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Convergence')
    plt.grid(True)
    plt.show()

    print(f"Training took {elapsed_time:.2f} seconds.")
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")

if __name__ == "__main__":
    main()