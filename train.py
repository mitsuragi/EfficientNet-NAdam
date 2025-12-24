import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score 
from tqdm import tqdm
import os

from model import EfficientNet
from dataset import get_dataloaders

def train(model, dataloader, optim, scheduler, criterion, epoch, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optim.zero_grad()
    pbar = tqdm(dataloader, desc=f'Train epoch {epoch}')
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        loss.backward()
        scheduler.step()
        optim.step()
        optim.zero_grad()

        total_loss += loss.item()
        _, preds = torch.max(pred, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Evaluation'):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item()
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Testing'):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1

def main():
    EPOCHS = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется {device}')

    os.makedirs('models', exist_ok=True)

    print('Загрузка данных\n')
    train_dataloader, eval_dataloader, test_dataloader, num_classes = get_dataloaders('input/', batch_size=64, img_size=224, num_workers=2)

    model = EfficientNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        anneal_strategy='cos'
    )

    print('Начало обучения')
    best_val_acc = 0
    train_history = []
    val_history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            model, train_dataloader, optimizer, scheduler, criterion, epoch, device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = eval(
            model, eval_dataloader, criterion, device
        )

        train_history.append({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1
        })

        val_history.append({
            'epoch': epoch,
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1
        })

        print(f'\nEpoch {epoch}/{EPOCHS}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
              f'Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, '
              f'Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = 'efficientnet_adam.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
            }, os.path.join('models', model_name))
            print(f'  Сохранена лучшая модель (Val Acc: {val_acc:.4f})')

    print('Тестировние лучшей модели')
    checkpoint = torch.load(os.path.join('models', model_name), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, test_prec, test_rec, test_f1 = test(model, test_dataloader, device)

    print(f'\n=== Результаты на тестовой выборке ===')
    print(f'Accuracy:  {test_acc:.4f}')
    print(f'Precision: {test_prec:.4f}')
    print(f'Recall:    {test_rec:.4f}')
    print(f'F1-score:  {test_f1:.4f}')
    
    # Сохранение результатов
    results = {
        # 'config': vars(args),
        'train_history': train_history,
        'val_history': val_history,
        'test_results': {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1
        }
    }
    
    import json
    results_name = 'efficientnet_adam.json'
    with open(os.path.join('models', results_name), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'\nРезультаты сохранены в {os.path.join('models', results_name)}')


if __name__ == '__main__':
    main()
