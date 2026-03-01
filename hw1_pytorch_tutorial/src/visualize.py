from matplotlib import pyplot as plt
import json
import numpy as np
from torchvision.transforms.functional import to_grayscale

with open("training_results.json", "r") as f:
    training_result_dict = json.load(f)

def create_arrays_from_list_of_dicts(list_of_dicts):
    result = {}
    for d in list_of_dicts:
        for key in d:
            try:
                result[key].append(d[key])
            except:
                result[key] = [d[key]]
    for key in result:
        result[key] = np.array(result[key])
    return result

processsed_training_results = {}
for k in training_result_dict:
    processsed_training_results[k] = create_arrays_from_list_of_dicts(training_result_dict[k])



# loss vs batch
ax = plt.axes()
lines=[]
for k in processsed_training_results:
    data = processsed_training_results[k]["train_loss"]
    x = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    line, = ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())
    lines.append(line)

ax.legend(lines,processsed_training_results.keys())
plt.title("Training Loss vs Batch\n10 Run Mean and Std Dev")
ax.set_xlabel("Batch")
ax.set_ylabel("Training Loss")
plt.show()

# test_accuracy vs epoch
ax = plt.axes()
lines=[]
for k in processsed_training_results:
    data = processsed_training_results[k]["test_accuracy"]
    x = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    line, = ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())
    lines.append(line)

ax.legend(lines, processsed_training_results.keys())
plt.title("Test Accuracy vs Epoch\n10 Run Mean and Std Dev")
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy (%)")
plt.show()

threshes = [2.0, 1.0, 0.5, 0.25, 0.1]
for thresh in threshes:
    for kind in training_result_dict:
        for i, loss in enumerate(training_result_dict[kind]["train_loss"]):
            if loss <= thresh:
                print(f"{kind} reached loss {thresh} at batch {i}")
                break
    print("---------------")
