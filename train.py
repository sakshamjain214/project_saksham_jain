# train.py
import torch
import time
import os

def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    # Execute on M4 GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    train_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Time: {epoch_mins}m {epoch_secs}s")
        
    print("Training complete! Saving model weights...")
    os.makedirs('_checkpoints', exist_ok=True)
    save_path = '_checkpoints/_final_weights.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Weights successfully saved to {save_path}")
        
    return model, train_losses