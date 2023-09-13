import re
import matplotlib.pyplot as plt

# 로그 파일 경로
log_file = "/root/License-Plate-Landmarks-detection/training_log.txt"

# 추출할 정보를 저장할 리스트
epochs = []
validation_losses = []
valid_loc_values = []
valid_cla_values = []
valid_landm_values = []

train_losses = []
train_loc_values = []
train_cla_values = []
train_landm_values = []

epoch_train_losses = []
epoch_train_loc_values = []
epoch_train_cla_values = []
epoch_train_landm_values = []


# 로그 파일을 한 줄씩 읽어서 정보 추출
with open(log_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        match = re.search(r"Validation Epoch:(\d+)/\d+ \|\| Loss: (\d+\.\d+) Loc: (\d+\.\d+) Cla: (\d+\.\d+) Landm: (\d+\.\d+)", line)
        if match:
            epoch, validation_loss, val_loc, va_cla, val_landm = map(float, match.groups())
            epochs.append(epoch)
            validation_losses.append(validation_loss)
            valid_loc_values.append(val_loc)
            valid_cla_values.append(va_cla)
            valid_landm_values.append(val_landm)

        match = re.search(r"Epochiter: (\d+)/(\d+) \|\| Iter: (\d+)/\d+ \|\| Loc: (\d+\.\d+) Cla: (\d+\.\d+) Landm: (\d+\.\d+)", line)
        if match:
            epochIter, epoch_max_iter, iter, loc, cla, landm = map(float, match.groups())
            epoch_train_losses.append(2.0 * loc + cla + landm)
            epoch_train_loc_values.append(loc)
            epoch_train_cla_values.append(cla)
            epoch_train_landm_values.append(landm)
            if (epochIter == epoch_max_iter) :
                train_losses.append(sum(epoch_train_losses) / (epoch_max_iter - 1))
                train_loc_values.append(sum(epoch_train_loc_values) / (epoch_max_iter - 1))
                train_cla_values.append(sum(epoch_train_cla_values) / (epoch_max_iter - 1))
                train_landm_values.append(sum(epoch_train_landm_values) / (epoch_max_iter - 1))
                epoch_train_losses.clear()
                epoch_train_loc_values.clear()
                epoch_train_cla_values.clear()
                epoch_train_landm_values.clear()

# 그래프 그리기
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss" + "[" + str(round(train_losses[-1], 2)) + "]")
plt.plot(epochs, validation_losses, label="Validation Loss" + "[" + str(round(validation_losses[-1], 2)) + "]")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_loc_values, label="Train Loc Loss" + "[" + str(round(train_losses[-1], 2)) + "]")
plt.plot(epochs, valid_loc_values, label="Validation Loc Loss"+ "[" + str(round(train_losses[-1], 2)) + "]")
plt.xlabel("Epoch")
plt.ylabel("Loc Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, train_cla_values, label="Train Cla Loss"+ "[" + str(round(train_cla_values[-1], 2)) + "]")
plt.plot(epochs, valid_cla_values, label="Validation Cla Loss"+ "[" + str(round(valid_cla_values[-1], 2)) + "]")
plt.xlabel("Epoch")
plt.ylabel("Classification Loss")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, train_landm_values, label="Train Landm Loss"+ "[" + str(round(train_landm_values[-1], 2)) + "]")
plt.plot(epochs, valid_landm_values, label="Validation Landm Loss"+ "[" + str(round(valid_landm_values[-1], 2)) + "]")
plt.xlabel("Epoch")
plt.ylabel("Landm Loss")
plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_loc_values, label="Loc")
# plt.plot(epochs, train_cla_values, label="Cla")
# plt.plot(epochs, train_landm_values, label="Landm")
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.legend()

plt.tight_layout()
plt.savefig('/root/License-Plate-Landmarks-detection/logs/loss_graph.jpg')
plt.show()
