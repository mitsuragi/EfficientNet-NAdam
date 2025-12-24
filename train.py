import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import precision_recall_fscore_support, accuracy_score 
from tqdm import tqdm
import os
import argparse
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
        optim.step()
        scheduler.step()
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
    parser = argparse.ArgumentParser(description='EfficientNet learning')
    parser.add_argument('--data_dir', type=str, default='input', help='Директория с исходными данными')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'nadam'], help='Оптимизатор (Adam или NAdam)')
    parser.add_argument('--pretrained', action='store_true',help='Использовать предобученную модель')
    parser.add_argument('--save_dir', type=str, default='models', help='Директория для сохранения моделей')
    parser.add_argument('--num_workers', type=int, default=0, help='Количество workers для Dataloader')

    args = parser.parse_args()

    print(vars(args))
    return

    if args.optimizer == 'nadam' and not args.pretrained:
        print('Так как используется не предобученная модель, то оптимизатор принудительно будет установлен Adam')

    IMG_SIZE = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    print('Загрузка данных\n')
    train_dataloader, eval_dataloader, test_dataloader, num_classes = get_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=IMG_SIZE, num_workers=args.num_workers)

    if (args.pretrained):
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(device)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes).to(device)
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = EfficientNet(num_classes).to(device)

    if args.optimizer == 'nadam' and args.pretrained:
        optimizer = optim.NAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        steps_per_epoch=len(train_dataloader),
        epochs=args.epochs,
        anneal_strategy='cos'
    )

    print(f'Начало обучения\tОптимизатор: {args.optimizer.upper()}, lr={args.lr}')
    best_val_acc = 0
    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
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

        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
              f'Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, '
              f'Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f'efficientnet_pretrained-{args.pretrained}_{args.optimizer}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
            }, os.path.join('models', model_name))
            print(f'  Сохранена лучшая модель (Val Acc: {val_acc:.4f})\n')

    print('Тестировние лучшей модели')
    checkpoint = torch.load(os.path.join(args.save_dir, model_name), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, test_prec, test_rec, test_f1 = test(model, test_dataloader, device)

    print(f'\n=== Результаты на тестовой выборке ===')
    print(f'Accuracy:  {test_acc:.4f}')
    print(f'Precision: {test_prec:.4f}')
    print(f'Recall:    {test_rec:.4f}')
    print(f'F1-score:  {test_f1:.4f}')
    
    # Сохранение результатов
    results = {
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
    results_name = f'efficientnet_pretrained_{args.pretrained}_{args.optimizer}.json'
    with open(os.path.join(args.save_dir, results_name), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'\nРезультаты сохранены в {os.path.join(args.save_dir, results_name)}')


if __name__ == '__main__':
    main()
