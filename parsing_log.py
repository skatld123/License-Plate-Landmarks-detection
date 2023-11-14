import matplotlib.pyplot as plt
import re

# 로그 파일을 읽고 데이터를 파싱하는 함수
def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    train_data = {'epoch': [], 'loc': [], 'cla': [], 'landm': []}
    val_data = {'epoch': [], 'loss': [], 'loc': [], 'cla': [], 'landm': []}

    for line in lines:
        if 'Validation' in line:
            match = re.search(r'Epoch:(\d+)/\d+.*?Loss: (\d+\.\d+).*?Loc: (\d+\.\d+).*?Cla: (\d+\.\d+).*?Landm: (\d+\.\d+)', line)
            if match:
                epoch, loss, loc, cla, landm = map(float, match.groups())
                val_data['epoch'].append(epoch)
                val_data['loss'].append(loss)
                val_data['loc'].append(loc)
                val_data['cla'].append(cla)
                val_data['landm'].append(landm)
        else:
            match = re.search(r'Epoch:(\d+)/\d+.*?Loc: (\d+\.\d+).*?Cla: (\d+\.\d+).*?Landm: (\d+\.\d+)', line)
            if match:
                epoch, loc, cla, landm = map(float, match.groups())
                train_data['epoch'].append(epoch)
                train_data['loc'].append(loc)
                train_data['cla'].append(cla)
                train_data['landm'].append(landm)

    return train_data, val_data

# 그래프를 그리는 함수
def plot_data(train_data, val_data):
    plt.figure(figsize=(12, 8))

    # Train 데이터 그래프
    plt.subplot(2, 1, 1)
    plt.plot(train_data['epoch'], train_data['loc'], label='Loc')
    plt.plot(train_data['epoch'], train_data['cla'], label='Cla')
    plt.plot(train_data['epoch'], train_data['landm'], label='Landm')
    plt.title('Train Data')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    # Validation 데이터 그래프
    plt.subplot(2, 1, 2)
    plt.plot(val_data['epoch'], val_data['loss'], label='Loss')
    plt.plot(val_data['epoch'], val_data['loc'], label='Loc')
    plt.plot(val_data['epoch'], val_data['cla'], label='Cla')
    plt.plot(val_data['epoch'], val_data['landm'], label='Landm')
    plt.title('Validation Data')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/root/deIdentification-clp/clp_landmark_detection/logs/results.jpg')
    plt.show()

# 평균을 계산하는 함수
def calculate_averages(data):
    avg_loc = sum(data['loc']) / len(data['loc'])
    avg_cla = sum(data['cla']) / len(data['cla'])
    avg_landm = sum(data['landm']) / len(data['landm'])
    return avg_loc, avg_cla, avg_landm

# 로그 파일을 파싱합니다.
log_file_path = '/root/deIdentification-clp/clp_landmark_detection/logs/training_log.txt'
train_data, val_data = parse_log_file(log_file_path)

# 그래프를 그립니다.
plot_data(train_data, val_data)

# 평균을 계산합니다.
avg_loc, avg_cla, avg_landm = calculate_averages(train_data)
print(f"Train Data Averages - Loc: {avg_loc}, Cla: {avg_cla}, Landm: {avg_landm}")
