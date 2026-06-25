import os
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import *
from data_utils import get_dataloaders
from model import BertNER


def compute_metrics(predictions, labels, masks, id_to_label):
    tp = {}
    fp = {}
    fn = {}
    
    for label in id_to_label.values():
        tp[label] = 0
        fp[label] = 0
        fn[label] = 0
    
    for pred, label, mask in zip(predictions, labels, masks):
        if isinstance(mask, list):
            actual_len = sum(mask)
        else:
            actual_len = int(mask.sum().item())
        pred = pred[:actual_len]
        label = label[:actual_len]
        
        for p, l in zip(pred, label):
            p_label = id_to_label[p]
            l_label = id_to_label[l]
            
            if p_label == l_label:
                tp[p_label] += 1
            else:
                fp[p_label] += 1
                fn[l_label] += 1
    
    precision = {}
    recall = {}
    f1 = {}
    
    for label in id_to_label.values():
        if tp[label] + fp[label] == 0:
            precision[label] = 0.0
        else:
            precision[label] = tp[label] / (tp[label] + fp[label])
        
        if tp[label] + fn[label] == 0:
            recall[label] = 0.0
        else:
            recall[label] = tp[label] / (tp[label] + fn[label])
        
        if precision[label] + recall[label] == 0:
            f1[label] = 0.0
        else:
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
    
    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    return precision, recall, f1, micro_f1


def train(model, train_loader, dev_loader, optimizer, scheduler, device, epochs, patience, model_dir):
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_masks = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_masks.extend(attention_mask.cpu().numpy().tolist())
        
        _, _, f1, micro_f1 = compute_metrics(all_predictions, all_labels, all_masks, ID_TO_LABEL)
        print(f"Dev Micro F1: {micro_f1:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'loss': avg_loss
        }
        torch.save(checkpoint, os.path.join(model_dir, 'last_checkpoint.pt'))
        
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            patience_counter = 0
            torch.save(checkpoint, os.path.join(model_dir, 'best_model.pt'))
            print(f"New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_loader, dev_loader, test_loader, tokenizer = get_dataloaders(BATCH_SIZE, MAX_LEN)
    
    model = BertNER(
        num_labels=NUM_LABELS,
        model_name=BERT_MODEL_NAME
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    start_epoch = 0
    best_f1 = 0.0
    
    checkpoint_path = os.path.join(MODEL_DIR, 'last_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        print(f"Resuming training from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    train(model, train_loader, dev_loader, optimizer, scheduler, DEVICE, EPOCHS, PATIENCE, MODEL_DIR)


if __name__ == '__main__':
    main()