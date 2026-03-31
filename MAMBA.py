#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import numpy as np
import pandas as pd

base_path = r"C:\Users\hp\Desktop\ZJUT\My Thesis\mm-fit"
participants = [f"w{str(i).zfill(2)}" for i in range(21)]  # w00 to w20

modalities = {
    "eb_l_acc": "eb_l_acc.npy",    # elbow motion
    "eb_l_gyr": "eb_l_gyr.npy",    # elbow gyroscope
    "sw_l_hr": "sw_l_hr.npy",      # heart rate (audio proxy)
    "pose_2d":  "pose_2d.npy",     # 2D pose
    "pose_3d":  "pose_3d.npy"      # 3D pose
}

results = []

print("🟩 Starting data loading...\n")

for w in participants:
    w_path = os.path.join(base_path, w)
    if not os.path.isdir(w_path):
        print(f"❌ Participant folder not found: {w_path}")
        continue

    participant_row = {"participant": w}
    for key, filename in modalities.items():
        file_path = os.path.join(w_path, f"{w}_{filename}")
        try:
            data = np.load(file_path)
            participant_row[key] = data.shape
            print(f"✅ Loaded {file_path} | shape = {data.shape}")
        except FileNotFoundError:
            print(f"⚠️ Missing file: {file_path}")
            participant_row[key] = None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            participant_row[key] = "error"

    results.append(participant_row)

print("\n✅ Done loading. Creating DataFrame...\n")

df = pd.DataFrame(results)
from IPython.display import display
display(df)


# In[2]:


import pandas as pd

def load_labels(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"⚠️ Labels file not found: {path}")
        return None


# In[3]:


labels = load_labels(r"C:\Users\hp\Desktop\ZJUT\My Thesis\mm-fit\w00\w00_labels.csv")

# If loaded successfully, print first 5 rows
if labels is not None:
    print(labels.head())


# In[4]:


labels = pd.read_csv(
    r"C:\Users\hp\Desktop\ZJUT\My Thesis\mm-fit\w00\w00_labels.csv",
    header=None,  # No header in file
    names=["start", "end", "label_id", "label_name"]  # Assign your own
)

print(labels.head())


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_modality(path):
    try:
        return np.load(path)
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return None

def load_labels(path):
    try:
        return pd.read_csv(path, header=None, names=["start", "end", "label_id", "label_name"])
    except FileNotFoundError:
        print(f"⚠️ Labels file not found: {path}")
        return None

def plot_sensor_with_labels(data, labels, data_type="acc", title=""):
    if data is None:
        print("⚠️ No data to plot.")
        return

    plt.figure(figsize=(14, 4))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=f"{data_type}_{i}")

    for _, row in labels.iterrows():
        start, end, _, label_name = row
        plt.axvspan(start, end, color='orange', alpha=0.3, label=label_name)

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel(data_type.upper())
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()


# In[9]:


print("Shape of sw_l_acc:", modalities['sw_l_acc'].shape)
print("Sample values:\n", modalities['sw_l_acc'][:5])


# In[10]:


print("Max index in labels:", labels['end'].max())


# In[12]:


acc_raw = modalities['sw_l_acc']         # (220297, 5)
acc_clean = acc_raw[:, 2:]               # Only columns with real sensor data
plot_sensor_with_labels(acc_clean, labels, data_type='acc', title='Accelerometer with Activity Labels')


# In[13]:


def plot_heart_rate(hr_data, title='Heart Rate (BPM)'):
    if hr_data is None or hr_data.shape[1] < 3:
        print("Invalid heart rate data")
        return

    time = np.arange(hr_data.shape[0])
    bpm = hr_data[:, 2]  # Use column 2 (likely BPM)

    plt.figure(figsize=(10, 4))
    plt.plot(time, bpm, color='red')
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('HR (bpm)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot corrected HR
plot_heart_rate(modalities['sw_l_hr'], title='Smartwatch Left Heart Rate (Corrected)')


# In[15]:


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Configuration ===
base_dir = r"C:\Users\hp\Desktop\ZJUT\Thesis\mm-fit\w00"
frame = 53000

# === Load pose data ===
pose_2d = np.load(os.path.join(base_dir, "w00_pose_2d.npy"))  # shape: (2, num_frames, 19)
pose_3d = np.load(os.path.join(base_dir, "w00_pose_3d.npy"))  # shape: (3, num_frames, 18)

# === Slice only 17 valid joints ===
skel_2d = pose_2d[:, frame, 1:18]  # shape: (2, 17)
skel_3d = pose_3d[:, frame, 1:18]  # shape: (3, 17)

# === COCO-style limb connections ===
connections_2d = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # right arm
    (1, 5), (5, 6), (6, 7),                 # left arm
    (1, 8), (8, 9), (9, 10),                # right leg
    (8, 11), (11, 12), (12, 13),            # left leg
    (0, 14), (14, 16), (0, 15)              # head and eyes (removed (15, 17) to avoid IndexError)
]

connections_3d = [
    (0, 1), (1, 2), (2, 3),                 # left leg
    (0, 4), (4, 5), (5, 6),                 # right leg
    (0, 7), (7, 8), (8, 9), (9, 10),        # spine to head
    (0, 11), (11, 12), (12, 13),            # left arm
    (11, 14), (14, 15), (15, 16)            # right arm
]

# === Plot 2D Pose ===
plt.figure(figsize=(8, 10))
plt.title("2D Pose Estimate")
plt.gca().invert_yaxis()
plt.scatter(skel_2d[0], skel_2d[1], c='red')
for i, j in connections_2d:
    plt.plot([skel_2d[0][i], skel_2d[0][j]], [skel_2d[1][i], skel_2d[1][j]], 'k-')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# === Plot 3D Pose ===
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Pose Estimate")
ax.scatter(skel_3d[0], skel_3d[1], skel_3d[2], c='red')
for i, j in connections_3d:
    ax.plot([skel_3d[0][i], skel_3d[0][j]],
            [skel_3d[1][i], skel_3d[1][j]],
            [skel_3d[2][i], skel_3d[2][j]], 'k-')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()


# In[16]:


import os
import numpy as np

# Step 1: Setup
base_path = r"C:\Users\hp\Desktop\ZJUT\Thesis\mm-fit"
participants = [f"w{str(i).zfill(2)}" for i in range(21)]
modalities_to_load = [
    "sw_l_acc",     # Smartwatch left accelerometer
    "sp_r_mag",     # Smartphone right magnetometer
    "eb_l_gyr",     # Earbud left gyroscope
    "pose_2d",      # 2D Pose
    "pose_3d",      # 3D Pose
    "sw_l_hr",      # Smartwatch left heart rate
]

# Function to load a single file, safely
def try_load(path):
    try:
        return np.load(path)
    except FileNotFoundError:
        return None

# Step 2: Load all selected modalities
participant_data = {}
for p in participants:
    data = {}
    participant_path = os.path.join(base_path, p)
    all_loaded = True
    for mod in modalities_to_load:
        file_name = f"{p}_{mod}.npy"
        file_path = os.path.join(participant_path, file_name)
        mod_data = try_load(file_path)
        if mod_data is None:
            all_loaded = False
        data[mod] = mod_data
    if all_loaded:
        participant_data[p] = data

# Step 3: Summary
print(f"✅ Valid participants with all selected modalities: {len(participant_data)}")
print(f"Participants: {list(participant_data.keys())}")


# In[4]:


import os
import numpy as np

# Define base path to your dataset
base_dir = r"C:\Users\hp\Desktop\ZJUT\Thesis\mm-fit"

# List of selected modalities
modalities = ['sw_l_acc', 'sp_r_mag', 'eb_l_gyr', 'pose_2d', 'pose_3d', 'sw_l_hr']

# List of 18 valid participants (already confirmed)
valid_participants = ['w00', 'w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 'w09',
                      'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w19', 'w20']

# Dictionary to store loaded data
participant_data = {}

# Loop through participants
for pid in valid_participants:
    data = {}
    participant_path = os.path.join(base_dir, pid)
    all_found = True

    for modality in modalities:
        fname = f"{pid}_{modality}.npy"
        fpath = os.path.join(participant_path, fname)

        try:
            data[modality] = np.load(fpath)
        except FileNotFoundError:
            print(f"❌ Missing: {fpath}")
            all_found = False
            break

    if all_found:
        participant_data[pid] = data

print(f"\n✅ Loaded data for {len(participant_data)} participants with all selected modalities.")
print(f"Participants: {list(participant_data.keys())}")


# In[5]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define path and config
base_path = r"C:\Users\hp\Desktop\ZJUT\Thesis\mm-fit"
participants = [
    'w00', 'w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 'w09',
    'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w19', 'w20'
]
modalities = ['sw_l_acc', 'sp_r_mag', 'eb_l_gyr', 'pose_2d', 'pose_3d', 'sw_l_hr']

# Load data
data_dict = {}
shape_records = []

for pid in tqdm(participants, desc="🔄 Loading participant data"):
    participant_data = {}
    for modality in modalities:
        file_path = os.path.join(base_path, pid, f"{pid}_{modality}.npy")
        try:
            data = np.load(file_path)
            participant_data[modality] = data
            shape_records.append({
                "Participant": pid,
                "Modality": modality,
                "Shape": data.shape
            })
        except FileNotFoundError:
            print(f"❌ Missing: {file_path}")
            participant_data[modality] = None
            shape_records.append({
                "Participant": pid,
                "Modality": modality,
                "Shape": None
            })
    data_dict[pid] = participant_data

# Show as DataFrame
df_shapes = pd.DataFrame(shape_records)
df_shapes_pivot = df_shapes.pivot(index="Participant", columns="Modality", values="Shape")
display(df_shapes_pivot)


# In[18]:


for pid in participants:
    labels = data_dict[pid]["labels"]
    if labels is not None:
        print(f"{pid}: {len(labels)} labels")
    else:
        print(f"{pid}: ❌ No label file")


# In[19]:


for pid in participants:
    print(f"\n📋 Labels for {pid}:")
    labels = data_dict[pid]["labels"]
    if labels is not None:
        print(labels.to_string(index=False))
    else:
        print("⚠️ No labels found.")


# In[20]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define base path and configuration
base_path = r"C:\Users\hp\Desktop\ZJUT\Thesis\mm-fit"
participants = [
    'w00', 'w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 'w09',
    'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w19', 'w20'
]
modalities = ['sw_l_acc', 'sp_r_mag', 'eb_l_gyr', 'pose_2d', 'pose_3d', 'sw_l_hr']

# Storage
data_dict = {}
shape_records = []

# Load participant data
for pid in tqdm(participants, desc="🔄 Loading participant data"):
    participant_data = {}
    
    # Load modalities
    for modality in modalities:
        file_path = os.path.join(base_path, pid, f"{pid}_{modality}.npy")
        try:
            data = np.load(file_path)
            participant_data[modality] = data
            shape_records.append({
                "Participant": pid,
                "Modality": modality,
                "Shape": data.shape
            })
        except FileNotFoundError:
            print(f"❌ Missing: {file_path}")
            participant_data[modality] = None
            shape_records.append({
                "Participant": pid,
                "Modality": modality,
                "Shape": None
            })

    # ✅ Load label CSV
    label_path = os.path.join(base_path, pid, f"{pid}_labels.csv")
    try:
        labels = pd.read_csv(label_path, header=None, names=["start", "end", "label_id", "label_name"])
        participant_data["labels"] = labels
        print(f"📋 {pid}: Loaded {len(labels)} labels")
    except FileNotFoundError:
        print(f"⚠️ {pid}: Missing label file")
        participant_data["labels"] = None

    data_dict[pid] = participant_data

# Convert shape records to DataFrame for inspection (optional)
df_shapes = pd.DataFrame(shape_records)
df_shapes_pivot = df_shapes.pivot(index="Participant", columns="Modality", values="Shape")
display(df_shapes_pivot)


# In[28]:


from sklearn.preprocessing import LabelEncoder

# Encode labels to continuous integers starting from 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# You can later decode with:
# label_encoder.inverse_transform(preds)


# In[5]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configuration
base_path = r"C:\Users\hp\Desktop\ZJUT\Thesis\mm-fit"
modalities = ["sw_l_acc", "sp_r_mag", "eb_l_gyr", "pose_2d", "pose_3d"]
window_size = 128
stride = 64

X = []
y = []

# Loop through each participant
for pid in tqdm(os.listdir(base_path), desc="📦 Preparing data"):
    participant_path = os.path.join(base_path, pid)
    if not os.path.isdir(participant_path):
        continue

    # Load labels
    label_path = os.path.join(participant_path, f"{pid}_labels.csv")
    try:
        labels_df = pd.read_csv(label_path, header=None, names=["start", "end", "label_id", "label_name"])
    except:
        continue

    # Load all modalities
    modality_data = {}
    for mod in modalities:
        mod_path = os.path.join(participant_path, f"{pid}_{mod}.npy")
        if os.path.exists(mod_path):
            modality_data[mod] = np.load(mod_path)
        else:
            modality_data[mod] = None

    # Extract labeled segments
    for _, row in labels_df.iterrows():
        start, end = int(row.start), int(row.end)
        label = row.label_name.strip()  # ✅ Use label_name, not label_id

        segment_valid = True
        segment_modalities = []

        for mod in modalities:
            data = modality_data[mod]
            if data is None or (mod in ["pose_2d", "pose_3d"] and data.shape[1] < end) or (mod not in ["pose_2d", "pose_3d"] and data.shape[0] < end):
                segment_valid = False
                break
            if mod in ["pose_2d", "pose_3d"]:
                segment_modalities.append(data[:, start:end])
            else:
                segment_modalities.append(data[start:end])

        if not segment_valid:
            continue

        # Stack all modalities into one tensor (early fusion)
        stacked = np.concatenate([m.reshape(end-start, -1) for m in segment_modalities], axis=-1)

        # Slice into fixed-size windows
        for i in range(0, stacked.shape[0] - window_size + 1, stride):
            window = stacked[i:i+window_size]
            X.append(window)
            y.append(label)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# ✅ Label Encoding (move this here from Cell 1)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
nsamples, ntimesteps, nfeatures = X.shape
X = scaler.fit_transform(X.reshape(-1, nfeatures)).reshape(nsamples, ntimesteps, nfeatures)


# Split the dataset
from collections import Counter

# Check label counts
label_counts = Counter(y)
min_count = min(label_counts.values())

# Use stratified split only if all classes have at least 2 samples
if min_count >= 2:
    stratify_flag = y
else:
    print("⚠️ Not enough samples in some classes, skipping stratification.")
    stratify_flag = None

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=stratify_flag, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp if min_count >= 2 else None, random_state=42)



print(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"✅ X_test:  {X_test.shape}, y_test:  {y_test.shape}")
# Save this for decoding predictions later
label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print(f"🗺️ Label mapping: {label_mapping}")
import matplotlib.pyplot as plt
from collections import Counter

original_y = label_encoder.inverse_transform(y)
label_counts = Counter(original_y)

plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel("Original Label ID")
plt.ylabel("Number of samples")
plt.title("Class Distribution After Windowing")
plt.show()


# In[6]:


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)

# Model setup
input_dim = X_train.shape[2]
hidden_dim = 64
output_dim = len(np.unique(y))  # number of classes

model = LSTMClassifier(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[7]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Define model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # get the last time step
        return self.fc(out)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create datasets and loaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Instantiate model
input_dim = X_train.shape[2]
hidden_dim = 128
output_dim = len(np.unique(y_train))  # number of activity classes
model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        model.train()

        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")


# In[8]:


train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Put model in eval mode
model.eval()

# Step 2: Gather predictions and ground truths
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Step 3: Inverse transform to original label IDs
original_preds = label_encoder.inverse_transform(all_preds)
original_labels = label_encoder.inverse_transform(all_labels)

# Step 4: Print classification report
print(classification_report(original_labels, original_preds))

# Step 5: Plot confusion matrix
cm = confusion_matrix(original_labels, original_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=np.unique(original_labels))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on Test Set (Original Labels)")
plt.tight_layout()
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Generate report again
report_dict = classification_report(original_labels, original_preds, output_dict=True)

# Get labels (excluding 'accuracy', 'macro avg', etc.)
labels = [label for label in report_dict.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]

# Extract precision, recall, f1
precision = [report_dict[label]["precision"] for label in labels]
recall = [report_dict[label]["recall"] for label in labels]
f1 = [report_dict[label]["f1-score"] for label in labels]

# Plot
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')

plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Early Fusion - LSTM)")
plt.legend()
plt.tight_layout()
plt.show()


# In[12]:


class FeatureFusionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, modality_dims):
        self.X = X  # shape: (num_samples, seq_len, total_features)
        self.y = y
        self.modalities = modality_dims
        self.splits = self._compute_splits()

    def _compute_splits(self):
        indices = [0]
        total = 0
        for d in self.modalities:
            total += d
            indices.append(total)
        return indices

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        # Split sample into separate modality tensors
        modality_inputs = {}
        for i, mod in enumerate(self.modalities):
            start = self.splits[i]
            end = self.splits[i+1]
            modality_inputs[f"mod{i}"] = torch.tensor(sample[:, start:end], dtype=torch.float32)
        return modality_inputs, torch.tensor(label, dtype=torch.long)


# In[13]:


class FeatureFusionLSTM(nn.Module):
    def __init__(self, modality_dims, hidden_dim, num_classes):
        super().__init__()
        self.modalities = nn.ModuleList([
            nn.LSTM(input_size=dim, hidden_size=hidden_dim, batch_first=True)
            for dim in modality_dims
        ])
        self.classifier = nn.Linear(hidden_dim * len(modality_dims), num_classes)

    def forward(self, x_dict):
        features = []
        for i, lstm in enumerate(self.modalities):
            out, _ = lstm(x_dict[f"mod{i}"])  # shape: (batch, seq_len, hidden)
            features.append(out[:, -1, :])    # get last timestep
        fused = torch.cat(features, dim=-1)
        return self.classifier(fused)


# In[14]:


# First, define the number of features in each modality (as used earlier):
modality_dims = [3*1, 3*1, 3*1, 34*1, 51*1]  # Adjust if dimensions differ

# Use the same X_train, y_train, etc. but change dataset:
train_dataset = FeatureFusionDataset(X_train, y_train, modality_dims)
val_dataset = FeatureFusionDataset(X_val, y_val, modality_dims)
test_dataset = FeatureFusionDataset(X_test, y_test, modality_dims)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)


# In[16]:


def train_feature_fusion_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            # ✅ Move all modalities in dict to device
            X_batch = {k: v.to(device) for k, v in X_batch.items()}
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        # Compute train accuracy
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = {k: v.to(device) for k, v in X_val_batch.items()}
                y_val_batch = y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        model.train()

        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {train_acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")


# In[18]:


input_dims = modality_dims
hidden_dim = 64
output_dim = len(np.unique(y_train))

feature_model = FeatureFusionLSTM(input_dims, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(feature_model.parameters(), lr=1e-3)

train_feature_fusion_model(feature_model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Put model in eval mode
feature_model.eval()

# Step 2: Gather predictions and ground truths
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # ✅ Move all modalities to device
        X_batch = {k: v.to(device) for k, v in X_batch.items()}
        y_batch = y_batch.to(device)

        outputs = feature_model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Step 3: Inverse transform to original label IDs
original_preds = label_encoder.inverse_transform(all_preds)
original_labels = label_encoder.inverse_transform(all_labels)

# Step 4: Print classification report
print(classification_report(original_labels, original_preds))

# Step 5: Plot confusion matrix
cm = confusion_matrix(original_labels, original_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=np.unique(original_labels))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on Test Set (Feature Fusion - Original Labels)")
plt.tight_layout()
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# ✅ Generate report again using feature fusion results
report_dict = classification_report(original_labels, original_preds, output_dict=True)

# ✅ Get activity labels only
labels = [label for label in report_dict.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]

# ✅ Extract metrics
precision = [report_dict[label]["precision"] for label in labels]
recall = [report_dict[label]["recall"] for label in labels]
f1 = [report_dict[label]["f1-score"] for label in labels]

# ✅ Plot
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')

plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Feature Fusion - LSTM)")
plt.legend()
plt.tight_layout()
plt.show()


# In[21]:


import torch.nn.functional as F

class LateFusionLSTM(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim):
        super(LateFusionLSTM, self).__init__()
        self.modalities = nn.ModuleDict({
            mod: nn.LSTM(input_dim, hidden_dim, batch_first=True)
            for mod, input_dim in input_dims.items()
        })
        self.classifiers = nn.ModuleDict({
            mod: nn.Linear(hidden_dim, output_dim)
            for mod in input_dims.keys()
        })
    
    def forward(self, x_dict):
        logits = []
        for mod, lstm in self.modalities.items():
            x = x_dict[mod]  # shape: (batch, seq, input_dim)
            _, (hn, _) = lstm(x)
            out = hn[-1]
            logit = self.classifiers[mod](out)
            logits.append(logit)
        # Average logits from each modality
        fused = torch.stack(logits).mean(dim=0)
        return fused


# In[23]:


input_dims = {
    "sw_l_acc": 5,
    "sp_r_mag": 5,
    "eb_l_gyr": 5,
    "pose_2d": 38,
    "pose_3d": 54,
}


# In[27]:


from sklearn.metrics import accuracy_score

def train_late_fusion_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch = {mod: x.to(device) for mod, x in X_batch.items()}
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # Fused output
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = {mod: x.to(device) for mod, x in X_val_batch.items()}
                y_val_batch = y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {train_acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")


# In[29]:


modality_dims = {
    "sw_l_acc": 5,
    "sp_r_mag": 5,
    "eb_l_gyr": 5,
    "pose_2d": 38,
    "pose_3d": 54,
}

# Offsets to slice X into modality-specific parts
offsets = {}
start = 0
for mod, dim in modality_dims.items():
    offsets[mod] = (start, start + dim)
    start += dim

# Now split X into dicts
def convert_to_modality_dict(X):
    out = {mod: X[:, :, start:end] for mod, (start, end) in offsets.items()}
    return out

# Split all sets
X_train_dict = convert_to_modality_dict(X_train)
X_val_dict = convert_to_modality_dict(X_val)
X_test_dict = convert_to_modality_dict(X_test)


# In[30]:


from torch.utils.data import Dataset

class DictDataset(Dataset):
    def __init__(self, X_dict, y):
        self.X_dict = X_dict
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {mod: torch.tensor(self.X_dict[mod][idx], dtype=torch.float32) for mod in self.X_dict}
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return sample, label


# In[31]:


from torch.utils.data import DataLoader

train_loader = DataLoader(DictDataset(X_train_dict, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(DictDataset(X_val_dict, y_val), batch_size=64)
test_loader = DataLoader(DictDataset(X_test_dict, y_test), batch_size=64)


# In[32]:


train_late_fusion_model(late_model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Evaluate
late_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = {mod: x.to(device) for mod, x in X_batch.items()}
        y_batch = y_batch.to(device)
        outputs = late_model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Step 2: Decode labels
original_preds = label_encoder.inverse_transform(all_preds)
original_labels = label_encoder.inverse_transform(all_labels)

# Step 3: Print report
print(classification_report(original_labels, original_preds))

# Step 4: Confusion matrix
cm = confusion_matrix(original_labels, original_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(original_labels))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Late Fusion - LSTM)")
plt.tight_layout()
plt.show()

# Step 5: Classification bar plot
report_dict = classification_report(original_labels, original_preds, output_dict=True)
labels = [label for label in report_dict.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]
precision = [report_dict[label]["precision"] for label in labels]
recall = [report_dict[label]["recall"] for label in labels]
f1 = [report_dict[label]["f1-score"] for label in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Late Fusion - LSTM)")
plt.legend()
plt.tight_layout()
plt.show()


# In[36]:


import matplotlib.pyplot as plt

# 🔢 Replace with your actual loss values for each epoch
loss_values = [
    109.1797, 85.3053, 77.0036, 73.6866, 72.4215, 
    70.3228, 70.1916, 72.8509, 72.3212, 70.6196
]

epochs = list(range(1, len(loss_values) + 1))

# 📊 Plot
plt.figure(figsize=(7, 4))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='steelblue')
plt.title("Loss Curve - Late Fusion + LSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[37]:


import matplotlib.pyplot as plt

# 🔢 Loss values for each fusion method
early_loss = [94.9643, 70.3132, 69.7826, 69.5823, 73.4609, 68.4936, 61.4965, 61.0245, 61.9471, 59.1805]
feature_loss = [179.9048, 138.5076, 129.7307, 123.8203, 122.2061, 123.3150, 120.6640, 117.2889, 113.9178, 118.7686]
late_loss = [109.1797, 85.3053, 77.0036, 73.6866, 72.4215, 70.3228, 70.1916, 72.8509, 72.3212, 70.6196]

epochs = list(range(1, 11))

# 📊 Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, early_loss, marker='o', label='Early Fusion - LSTM')
plt.plot(epochs, feature_loss, marker='s', label='Feature Fusion - LSTM')
plt.plot(epochs, late_loss, marker='^', label='Late Fusion - LSTM')

plt.title("📉 Loss Curves for Fusion Methods (LSTM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[38]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TinyMambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.in_proj(x)                         # (B, L, H)
        x = rearrange(x, 'b l d -> b d l')          # (B, H, L)
        x = self.conv1d(x)                          # (B, H, L)
        x = rearrange(x, 'b d l -> b l d')          # (B, L, H)
        x = F.gelu(x)
        x = self.norm(x)
        return self.out_proj(x)                     # (B, L, H)


# In[39]:


class MambaInspiredClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mamba = TinyMambaBlock(input_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mamba(x)               # (B, L, H)
        x = x.permute(0, 2, 1)          # (B, H, L)
        x = self.pool(x).squeeze(-1)    # (B, H)
        return self.classifier(x)       # (B, C)


# In[40]:


input_dim = X_train.shape[2]
hidden_dim = 64
output_dim = len(np.unique(y_train))

mamba_model = MambaInspiredClassifier(input_dim, hidden_dim, output_dim).to(device)


# In[42]:


# X_train_t should be a tensor, not a dict
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)


# In[44]:


from sklearn.metrics import accuracy_score

def train_mamba_early_fusion(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # 🔍 Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        model.train()

        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {train_acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")


# In[46]:


val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)


# In[47]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mamba_model.parameters(), lr=1e-3)

train_mamba_early_fusion(mamba_model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[50]:


# Sample values (replace with your actual values if saved)
train_acc_list = [0.5001, 0.5238, 0.5277, 0.5424, 0.5561, 0.5618, 0.5777, 0.5929, 0.5980, 0.6181]
val_acc_list =   [0.4972, 0.5056, 0.5321, 0.5140, 0.5084, 0.5503, 0.5461, 0.5866, 0.5712, 0.5712]
loss_list =      [63.7043, 60.6297, 59.4078, 57.8609, 56.4507, 55.7431, 54.2994, 52.0352, 51.6643, 49.8033]

epochs = range(1, len(train_acc_list) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc_list, marker='o', label='Train Accuracy')
plt.plot(epochs, val_acc_list, marker='o', label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("🔁 Accuracy Curve (Early Fusion - Mamba)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss_list, marker='o', color='red', label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("📉 Loss Curve (Early Fusion - Mamba)")
plt.legend()

plt.tight_layout()
plt.show()


# In[51]:


# Debug: Check the batch type and structure
for X_batch, y_batch in test_loader:
    print(f"Type of X_batch: {type(X_batch)}")
    if isinstance(X_batch, dict):
        print(f"Keys: {X_batch.keys()}")
    else:
        print(f"Shape: {X_batch.shape}")
    break


# In[52]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Evaluate Mamba early fusion model
mamba_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # Concatenate all modalities along the last dimension
        X_batch = torch.cat([X_batch[mod] for mod in ['sw_l_acc', 'sp_r_mag', 'eb_l_gyr', 'pose_2d', 'pose_3d']], dim=-1)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = mamba_model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Decode class names if using label encoder
original_preds = label_encoder.inverse_transform(all_preds)
original_labels = label_encoder.inverse_transform(all_labels)

# Print classification report
print(classification_report(original_labels, original_preds))

# Confusion matrix
cm = confusion_matrix(original_labels, original_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(original_labels))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Early Fusion - Mamba)")
plt.tight_layout()
plt.show()

# Bar plot for precision, recall, F1
report_dict = classification_report(original_labels, original_preds, output_dict=True)
labels = [label for label in report_dict.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]
precision = [report_dict[label]["precision"] for label in labels]
recall = [report_dict[label]["recall"] for label in labels]
f1 = [report_dict[label]["f1-score"] for label in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Early Fusion - Mamba)")
plt.legend()
plt.tight_layout()
plt.show()


# In[53]:


# 🔁 Replace these values with your real loss values from training
mamba_early_losses = [
    63.7043, 60.6297, 59.4078, 57.8609, 56.4507,
    55.7431, 54.2994, 52.0352, 51.6643, 49.8033
]

# 📈 Plot loss curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), mamba_early_losses, marker='o', label="Early Fusion - Mamba")
plt.title("📉 Loss Curve - Early Fusion + Mamba")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[55]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TinyMambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d l -> b l d')
        x = F.gelu(x)
        x = self.norm(x)
        return self.out_proj(x)

class FeatureFusionMamba(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim):
        super().__init__()
        self.encoders = nn.ModuleDict({
            mod: TinyMambaBlock(input_dim, hidden_dim) for mod, input_dim in input_dims.items()
        })
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim * len(input_dims), output_dim)

    def forward(self, x_dict):
        encoded = []
        for mod, encoder in self.encoders.items():
            x = x_dict[mod]
            x = encoder(x)
            x = x.permute(0, 2, 1)
            x = self.pool(x).squeeze(-1)
            encoded.append(x)
        fused = torch.cat(encoded, dim=-1)
        return self.classifier(fused)


# In[56]:


from sklearn.metrics import accuracy_score

def train_mamba_feature_fusion(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch = {k: v.to(device) for k, v in X_batch.items()}
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = {k: v.to(device) for k, v in X_val_batch.items()}
                y_val_batch = y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {train_acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")


# In[58]:


from torch.utils.data import Dataset

class DictDataset(Dataset):
    def __init__(self, X_dict, y):
        self.X_dict = X_dict
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {mod: torch.tensor(self.X_dict[mod][idx], dtype=torch.float32) for mod in self.X_dict}, torch.tensor(self.y[idx], dtype=torch.long)


# In[59]:


X_train_dict = convert_to_modality_dict(X_train)
X_val_dict = convert_to_modality_dict(X_val)
X_test_dict = convert_to_modality_dict(X_test)


# In[60]:


from torch.utils.data import DataLoader

train_loader = DataLoader(DictDataset(X_train_dict, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(DictDataset(X_val_dict, y_val), batch_size=64)
test_loader = DataLoader(DictDataset(X_test_dict, y_test), batch_size=64)


# In[61]:


train_mamba_feature_fusion(model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[62]:


import matplotlib.pyplot as plt

# Epochs and values from your training logs
epochs = list(range(1, 11))
train_acc = [0.4340, 0.5172, 0.5462, 0.5561, 0.5807, 0.5962, 0.6058, 0.6151, 0.6265, 0.6381]
val_acc = [0.4874, 0.5070, 0.5056, 0.5531, 0.5642, 0.5796, 0.5670, 0.5950, 0.6034, 0.6103]
loss = [75.9093, 61.6443, 57.6415, 55.9628, 53.6850, 51.5539, 50.8440, 49.3425, 46.9746, 45.7281]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='s')
plt.plot(epochs, loss, label="Loss", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("📊 Mamba + Feature Fusion Training Overview")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[63]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Labels
labels = ['bicep_curls', 'dumbbell_rows', 'dumbbell_shoulder_press', 'jumping_jacks',
          'lateral_shoulder_raises', 'lunges', 'pushups', 'situps', 'squats', 'tricep_extensions']

# Metrics (from your results)
precision = [0.29, 0.50, 0.71, 0.00, 0.50, 0.71, 0.56, 0.65, 0.79, 0.43]
recall =    [0.28, 0.39, 0.64, 0.00, 0.60, 0.76, 0.63, 0.66, 0.82, 0.43]
f1_score =  [0.28, 0.44, 0.67, 0.00, 0.55, 0.73, 0.59, 0.65, 0.80, 0.43]

# 📊 Bar chart for Precision / Recall / F1
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1_score, width=width, label='F1-Score')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Feature Fusion - Mamba)")
plt.legend()
plt.tight_layout()
plt.show()

# 🔲 Confusion Matrix (example)
cm = np.array([
    [19, 3, 2, 0, 7, 10, 5, 6, 12, 4],
    [2, 25, 4, 0, 6, 12, 6, 3, 3, 3],
    [1, 0, 51, 0, 9, 4, 4, 6, 3, 2],
    [0, 0, 0, 0, 2, 2, 1, 1, 1, 1],
    [3, 1, 6, 0, 47, 5, 6, 3, 4, 3],
    [2, 3, 1, 0, 3, 84, 5, 5, 5, 2],
    [4, 1, 4, 0, 3, 5, 38, 2, 1, 2],
    [3, 0, 2, 0, 3, 7, 2, 64, 12, 4],
    [2, 1, 2, 0, 1, 4, 1, 3, 63, 0],
    [4, 2, 3, 0, 4, 5, 3, 3, 5, 46],
])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("📘 Confusion Matrix (Feature Fusion - Mamba)")
plt.tight_layout()
plt.show()

# 📈 Epoch Curve
epochs = list(range(1, 11))
train_acc = [0.4340, 0.5172, 0.5462, 0.5561, 0.5807, 0.5962, 0.6058, 0.6151, 0.6265, 0.6381]
val_acc = [0.4874, 0.5070, 0.5056, 0.5531, 0.5642, 0.5796, 0.5670, 0.5950, 0.6034, 0.6103]
loss = [75.9093, 61.6443, 57.6415, 55.9628, 53.6850, 51.5539, 50.8440, 49.3425, 46.9746, 45.7281]

plt.figure(figsize=(12, 6))
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='s')
plt.plot(epochs, loss, label="Loss", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("📈 Mamba + Feature Fusion Training Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[64]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.in_proj(x)              # (B, L, H)
        x = x.permute(0, 2, 1)           # (B, H, L)
        x = self.conv1d(x)               # (B, H, L)
        x = x.permute(0, 2, 1)           # (B, L, H)
        x = F.gelu(x)
        x = self.norm(x)
        return self.out_proj(x)          # (B, L, H)

class LateFusionMamba(nn.Module):
    def __init__(self, input_dims: dict, hidden_dim: int, output_dim: int):
        super().__init__()
        self.modalities = nn.ModuleDict({
            mod: TinyMambaBlock(input_dim, hidden_dim)
            for mod, input_dim in input_dims.items()
        })
        self.classifiers = nn.ModuleDict({
            mod: nn.Linear(hidden_dim, output_dim)
            for mod in input_dims
        })

    def forward(self, x_dict):
        logits = []
        for mod, block in self.modalities.items():
            x = x_dict[mod]              # (B, L, D)
            x = block(x)                 # (B, L, H)
            x = x.mean(dim=1)            # (B, H)
            logit = self.classifiers[mod](x)  # (B, C)
            logits.append(logit)
        return torch.stack(logits).mean(dim=0)


# In[65]:


input_dims = {
    "sw_l_acc": 5,
    "sp_r_mag": 5,
    "eb_l_gyr": 5,
    "pose_2d": 38,
    "pose_3d": 54,
}
hidden_dim = 64
output_dim = len(np.unique(y_train))

late_mamba_model = LateFusionMamba(input_dims, hidden_dim, output_dim).to(device)


# In[66]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[68]:


modality_dims = {
    "sw_l_acc": 5,
    "sp_r_mag": 5,
    "eb_l_gyr": 5,
    "pose_2d": 38,
    "pose_3d": 54,
}


# In[69]:


class TinyMambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(x)
        x = self.norm(x)
        return self.out_proj(x)

class LateFusionMamba(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim):
        super().__init__()
        self.modalities = nn.ModuleDict({
            mod: TinyMambaBlock(input_dim, hidden_dim)
            for mod, input_dim in input_dims.items()
        })
        self.classifiers = nn.ModuleDict({
            mod: nn.Linear(hidden_dim, output_dim)
            for mod in input_dims
        })

    def forward(self, x_dict):
        logits = []
        for mod in x_dict:
            x = self.modalities[mod](x_dict[mod])         # (B, L, H)
            x = x.permute(0, 2, 1)                        # (B, H, L)
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)   # (B, H)
            logits.append(self.classifiers[mod](x))       # (B, C)
        return torch.stack(logits).mean(dim=0)            # (B, C)


# In[70]:


model = LateFusionMamba(input_dims=modality_dims, hidden_dim=64, output_dim=len(np.unique(y_train))).to(device)


# In[72]:


modality_dims = {
    "sw_l_acc": 5,
    "sp_r_mag": 5,
    "eb_l_gyr": 5,
    "pose_2d": 38,
    "pose_3d": 54,
}

model = LateFusionMamba(input_dims=modality_dims, hidden_dim=64, output_dim=len(np.unique(y_train))).to(device)


# In[73]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[74]:


train_mamba_feature_fusion(model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[75]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = {k: v.to(device) for k, v in X_batch.items()}
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Decode labels
original_preds = label_encoder.inverse_transform(all_preds)
original_labels = label_encoder.inverse_transform(all_labels)

# Report
print(classification_report(original_labels, original_preds))

# Confusion Matrix
cm = confusion_matrix(original_labels, original_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(original_labels))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Late Fusion - Mamba)")
plt.tight_layout()
plt.show()

# Classification Bar Plot
report_dict = classification_report(original_labels, original_preds, output_dict=True)
labels = [label for label in report_dict.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]
precision = [report_dict[label]["precision"] for label in labels]
recall = [report_dict[label]["recall"] for label in labels]
f1 = [report_dict[label]["f1-score"] for label in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Late Fusion - Mamba)")
plt.legend()
plt.tight_layout()
plt.show()


# In[76]:


import matplotlib.pyplot as plt

# Provided loss values from each epoch
losses = [82.0002, 64.1548, 60.1629, 57.9889, 55.0710, 53.0688, 51.5020, 49.7431, 47.8888, 46.8230]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses)+1), losses, marker='o', linestyle='-')
plt.title("📉 Training Loss Curve (Late Fusion - Mamba)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.xticks(range(1, len(losses)+1))
plt.tight_layout()
plt.show()


# In[133]:


import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)              # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]                # take last time step
        return self.fc(out)


# In[134]:


input_dim = X_train.shape[2]             # e.g., 2D pose = 34 or 51 features
hidden_dim = 64
output_dim = len(label_encoder_filtered.classes_)  # e.g., 9 classes if jumping_jacks removed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMClassifier(input_dim, hidden_dim, output_dim).to(device)


# In[135]:


import torch.optim as optim
from sklearn.metrics import accuracy_score

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)

def train_lstm_single_modality(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_acc_list, val_acc_list, loss_list = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        loss_list.append(total_loss)

        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {train_acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")

    return train_acc_list, val_acc_list, loss_list


# In[136]:


train_acc_list, val_acc_list, loss_list = train_lstm_single_modality(
    lstm_model, train_loader, val_loader, criterion, optimizer, epochs=10
)


# In[142]:


class MambaInspiredClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mamba = TinyMambaBlock(input_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mamba(x)               # (B, L, H)
        x = x.permute(0, 2, 1)          # (B, H, L)
        x = self.pool(x).squeeze(-1)    # (B, H)
        return self.classifier(x)       # (B, C)


# In[143]:


input_dim = X_train.shape[2]  # pose_2d features per timestep
hidden_dim = 64
output_dim = len(np.unique(y_train))

mamba_model = MambaInspiredClassifier(input_dim, hidden_dim, output_dim).to(device)


# In[144]:


train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)


# In[145]:


from sklearn.metrics import accuracy_score

def train_mamba_early_fusion(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        model.train()

        print(f"📘 Epoch {epoch+1}/{epochs} | 🏋️ Train Acc: {train_acc:.4f} | ✅ Val Acc: {val_acc:.4f} | 📉 Loss: {total_loss:.4f}")


# In[146]:


val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)


# In[147]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mamba_model.parameters(), lr=1e-3)

train_mamba_early_fusion(mamba_model, train_loader, val_loader, criterion, optimizer, epochs=10)


# In[148]:


epochs = range(1, len(train_acc_list) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc_list, marker='o', label='Train Accuracy')
plt.plot(epochs, val_acc_list, marker='o', label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("🔁 Accuracy Curve (Pose 2D - Mamba)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss_list, marker='o', color='red', label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("📉 Loss Curve (Pose 2D - Mamba)")
plt.legend()

plt.tight_layout()
plt.show()


# In[149]:


for X_batch, y_batch in test_loader:
    print(f"X_batch shape: {X_batch.shape}")
    break


# In[150]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

mamba_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = mamba_model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Decode to label names
original_preds = label_encoder_filtered.inverse_transform(all_preds)
original_labels = label_encoder_filtered.inverse_transform(all_labels)
label_names = [str(c) for c in label_encoder_filtered.classes_]

# Print classification report
print(classification_report(original_labels, original_preds, target_names=label_names))

# Confusion matrix
cm = confusion_matrix(original_labels, original_preds, labels=label_encoder_filtered.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Pose 2D - Mamba)")
plt.tight_layout()
plt.show()

# Bar plot
report_dict = classification_report(original_labels, original_preds, output_dict=True)
labels = [label for label in report_dict.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]
precision = [report_dict[label]["precision"] for label in labels]
recall = [report_dict[label]["recall"] for label in labels]
f1 = [report_dict[label]["f1-score"] for label in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Classification Metrics by Activity (Pose 2D - Mamba)")
plt.legend()
plt.tight_layout()
plt.show()

