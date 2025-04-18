import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Завантаження даних
df = pd.read_csv('tracks.csv')  # Замінити на ваш шлях до файлу tracks.csv

# Вибір ознак
features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]
target = "popularity"

# Видалення рядків з пропущеними значеннями
df = df[features + [target]].dropna()

# Перевірка типів даних
print("Data types before conversion:")
print(df.dtypes)

# Перетворення стовпців 'key' та 'mode' у числові значення
if df['key'].dtype == 'object':
    # Перетворення 'key' у числові значення
    key_mapping = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                  'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    df['key'] = df['key'].map(key_mapping)

# Перевірка типу даних після перетворення
if df['mode'].dtype == 'object':
    # Перетворення 'mode' у числові значення
    mode_mapping = {'Major': 1, 'Minor': 0}
    df['mode'] = df['mode'].map(mode_mapping)

# Обробляти будь-які значення NaN
df = df.dropna()

# Виведення типів даних після перетворення
X = df[features].values
y = df[target].values

# Нормалізація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Будова PyTorch DataLoader
torch_X_train = torch.tensor(X_train, dtype=torch.float32)
torch_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(torch_X_train, torch_y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Оприділення моделі
class MusicNet(nn.Module):
    def __init__(self):
        super(MusicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MusicNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Тренування моделі
losses = []

for epoch in range(50):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.4f}")

# Візуалізація втрат
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()

# Оцінка моделі
model.eval()
with torch.no_grad():
    torch_X_test = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(torch_X_test).numpy()

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print(f"Test MAE: {mae:.2f}")

