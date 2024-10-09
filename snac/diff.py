import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# Training loop (modifications)
num_epochs = 20  # Increased number of epochs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
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
        
        # ... (rest of the training loop remains the same)

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)  # Learning rate scheduling
    
    # ... (rest of the epoch logging remains the same)

# ... (rest of the script remains the same)