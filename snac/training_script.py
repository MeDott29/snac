import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import wandb

# Initialize wandb
wandb.init(project="snac-training", name="snac-experiment")

# Load the dataset
dataset = datasets.load_dataset("Alignment-Lab-AI/ttest")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextSNACDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['json']['text']
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        snac_0 = torch.zeros(4096)
        snac_0[item['snac_0']] = 1
        snac_1 = torch.zeros(4096)
        snac_1[item['snac_1']] = 1
        snac_2 = torch.zeros(4096)
        snac_2[item['snac_2']] = 1
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'snac_0': snac_0,
            'snac_1': snac_1,
            'snac_2': snac_2
        }

# Create datasets
train_dataset = TextSNACDataset(dataset['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ImprovedTextSNACModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, dropout_rate=0.2):
        super(ImprovedTextSNACModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 512, hidden_dim)
        
        self.snac_0_head = nn.Linear(hidden_dim, 4096)
        self.snac_1_head = nn.Linear(hidden_dim, 4096)
        self.snac_2_head = nn.Linear(hidden_dim, 4096)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        
        snac_0 = torch.sigmoid(self.snac_0_head(x))
        snac_1 = torch.sigmoid(self.snac_1_head(x))
        snac_2 = torch.sigmoid(self.snac_2_head(x))
        
        return snac_0, snac_1, snac_2

# Initialize model, loss, and optimizer
model = ImprovedTextSNACModel(tokenizer.vocab_size)
criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


# Log model architecture to wandb
wandb.watch(model)

# Dashboard setup
dashboard = widgets.Output()

def create_progress_bar(value, total, description):
    return f"""
    <div style="width:100%; background-color:#ddd; border-radius:5px;">
        <div style="width:{(value/total)*100}%; background-color:#4CAF50; height:20px; border-radius:5px;">
        </div>
    </div>
    <p>{description}: {value}/{total}</p>
    """

def update_dashboard(epoch, batch, total_batches, loss, loss_0, loss_1, loss_2, epoch_time, batch_time):
    with dashboard:
        clear_output(wait=True)
        display(HTML(f"""
        <h2>Training Dashboard</h2>
        <div style="display:flex; justify-content:space-between;">
            <div style="width:48%;">
                <h3>Overall Progress</h3>
                {create_progress_bar(epoch, num_epochs, "Epoch")}
                {create_progress_bar(batch, total_batches, "Batch")}
            </div>
            <div style="width:48%;">
                <h3>Loss Information</h3>
                <p>Total Loss: {loss:.4f}</p>
                <p>SNAC_0 Loss: {loss_0:.4f}</p>
                <p>SNAC_1 Loss: {loss_1:.4f}</p>
                <p>SNAC_2 Loss: {loss_2:.4f}</p>
            </div>
        </div>
        <div>
            <h3>Time Information</h3>
            <p>Epoch Time: {epoch_time:.2f}s</p>
            <p>Batch Time: {batch_time:.2f}s</p>
        </div>
        """))

# Display the dashboard
display(dashboard)

# Training loop
num_epochs = 10
print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        snac_0 = batch['snac_0']
        snac_1 = batch['snac_1']
        snac_2 = batch['snac_2']

        optimizer.zero_grad()
        
        pred_0, pred_1, pred_2 = model(input_ids, attention_mask)
        
        loss_0 = criterion(pred_0, snac_0)
        loss_1 = criterion(pred_1, snac_1)
        loss_2 = criterion(pred_2, snac_2)
        loss = loss_0 + loss_1 + loss_2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Added gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        
        batch_end_time = time.time()
        
        # Update dashboard and log to wandb every 10 batches
        if batch_idx % 10 == 0:
            update_dashboard(
                epoch + 1, 
                batch_idx, 
                len(train_loader), 
                loss.item(), 
                loss_0.item(), 
                loss_1.item(), 
                loss_2.item(),
                time.time() - epoch_start_time,
                batch_end_time - batch_start_time
            )
            wandb.log({
                "epoch": epoch + 1,
                "batch": batch_idx,
                "loss": loss.item(),
                "loss_0": loss_0.item(),
                "loss_1": loss_1.item(),
                "loss_2": loss_2.item(),
                "epoch_time": time.time() - epoch_start_time,
                "batch_time": batch_end_time - batch_start_time
            })

    epoch_end_time = time.time()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)  # Learning rate scheduling

    
    # Final update for the epoch
    update_dashboard(
        epoch + 1, 
        len(train_loader), 
        len(train_loader), 
        avg_loss, 
        loss_0.item(), 
        loss_1.item(), 
        loss_2.item(),
        epoch_end_time - epoch_start_time,
        batch_end_time - batch_start_time
    )
    
    # Log epoch summary to wandb
    wandb.log({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "epoch_time": epoch_end_time - epoch_start_time
    })

print("Training completed.")

# Save the model
torch.save(model.state_dict(), 'text_snac_model.pth')
print("Model saved as 'text_snac_model.pth'")

# Save the model to wandb
wandb.save('text_snac_model.pth')

# Finish the wandb run
wandb.finish()