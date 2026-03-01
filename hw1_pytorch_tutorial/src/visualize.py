from matplotlib import pyplot as plt
import json
import numpy as np
import scipy

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

thresholds = [1.5, 1.0, 0.5, 0.25]

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
    print(f"{k} Loss mean std dev: {np.mean(std):.4f}")
    for thresh in thresholds:
        for i, value in enumerate(mean):
            if value <= thresh:
                print(f"{k} reached loss {thresh} at batch {i}")
                break

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
    print(f"{k} Accuracy mean std dev: {np.mean(std):.4f}")

ax.legend(lines, processsed_training_results.keys())
plt.title("Test Accuracy vs Epoch\n10 Run Mean and Std Dev")
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy (%)")
plt.show()

ff_accs = processsed_training_results['ff']["test_accuracy"].T[-1]
geglu_accs = processsed_training_results['geglu']["test_accuracy"].T[-1]
reglu_accs = processsed_training_results['reglu']["test_accuracy"].T[-1]

print(scipy.stats.ttest_rel(ff_accs, geglu_accs))
print(scipy.stats.ttest_rel(ff_accs, reglu_accs))
