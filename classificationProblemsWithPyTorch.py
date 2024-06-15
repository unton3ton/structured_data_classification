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



dataset = pd.read_csv('iris.csv')
print(dataset.shape)
print(dataset.head())


X = dataset.drop(["variety"], axis = 1)
print(X.head())


y = dataset.filter(["variety"], axis = 1)
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

model = Net(input_size=4, hidden_size=8, num_classes=3)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


num_epochs = 100

loss_vals = []

for epoch in range(num_epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    loss_vals.append(loss.detach().numpy().item())

    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')




indexes = list(range(len(loss_vals)))
sns.lineplot(x = indexes, y = loss_vals)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss against epochs')
plt.show()


with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted)
    matrix = confusion_matrix(y_test, predicted)

    print(f'Accuracy: {100 * accuracy:.2f}%')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{matrix}')