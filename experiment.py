import subprocess

def run_command(cmd, desc):
    print(f'{desc}')
    print(f'Запуск {cmd}\n')

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode != 0:
        print(f'Ошибка при выполнении {desc}')
        return False
    else:
        print(f'Успешно {desc}')
        return True

def main():
    print('Обучение EfficientNet\n')

    experiments = [
        {
            'name': 'Дообучение модели с использованием Adam',
            'cmd': 'python train.py --pretrained --optimizer adam --epochs 30 --lr 1e-4 --batch_size 64',
            'save_name': 'pretrained_adam'
        },
        {
            'name': 'Дообучение модели с использованием NAdam',
            'cmd': 'python train.py --pretrained --optimizer nadam --epochs 30 --lr 1e-4 --batch_size 64',
            'save_name': 'pretrained_nadam'
        },
        {
            'name': 'Обучение модели с нуля с использованием Adam',
            'cmd': 'python train.py --optimizer adam --epochs 30 --lr 1e-4 --batch_size 64',
            'save_name': 'from_scratch_adam'
        }
    ]

    results_summary = []

    for i, exp in enumerate(experiments, 1):
        print(f'Эксперимент {i}/{len(experiments)}\n')

        success = run_command(exp['cmd'], exp['name'])
        results_summary.append({
            'name': exp['name'],
            'success': success
        })

    print('\nРезультаты:\n')

    for result in results_summary:
        status = 'Успешно' if result['success'] else 'Ошибка'
        print(f'{status}: {result['name']}')


if __name__ == '__main__':
    main()
