# https://wellsr.com/python/solving-classification-and-regression-problems-with-pytorch/

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



dataset = pd.read_csv('hearz.csv')
print(dataset.shape)
print(dataset.head())
print(dataset.info())
print(dataset.describe())


X = dataset.drop(["thal", "target"], axis = 1)
print(X.head())


y = dataset.filter(["target"], axis = 1)
print(y.head())


X = X.values
y = y.values.ravel()


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = Net(input_size=12, hidden_size=26, num_classes=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


num_epochs = 15

loss_vals = []

for epoch in range(num_epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    loss_vals.append(loss.detach().numpy().item())

    optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')




indexes = list(range(len(loss_vals)))
sns.lineplot(x = indexes, y = loss_vals)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss against epochs for herz-class')
plt.show()


with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    # print(predicted.shape)

    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted, zero_division=0)
    matrix = confusion_matrix(y_test, predicted)

    print(f'Accuracy: {100 * accuracy:.2f}%')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{matrix}')

with torch.no_grad():
    X = dataset.drop(["thal", "target"], axis = 1)
    print(X.shape)
    print(X.head())
    XtoTensor = torch.tensor(X.values).float()
    outputs = model(XtoTensor)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted.shape)
    X['target'] = dataset.filter(["target"], axis = 1)
    X['predicted'] = int(predicted[0])

    print(X.shape)
    # print(X.head(6))
    print(X[207:265])
    # print(X.tail())