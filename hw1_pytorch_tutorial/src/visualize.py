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

print(processsed_training_results)

'''

# loss vs batch
ax = plt.axes()
ff_line, = ax.plot(training_result_dict["ff"]["train_loss"])
reglu_line, = ax.plot(training_result_dict["reglu"]["train_loss"])
geglu_line, = ax.plot(training_result_dict["geglu"]["train_loss"])
ax.legend([ff_line, reglu_line, geglu_line],["ff", "reglu", "geglu"])
plt.title("Training Loss vs Batch")
ax.set_xlabel("Batch")
ax.set_ylabel("Training Loss")
plt.show()

# test_accuracy vs epoch
ax = plt.axes()
ff_line, = ax.plot(training_result_dict["ff"]["test_accuracy"], marker='o')
reglu_line, = ax.plot(training_result_dict["reglu"]["test_accuracy"], marker='o')
geglu_line, = ax.plot(training_result_dict["geglu"]["test_accuracy"], marker='o')
ax.legend([ff_line, reglu_line, geglu_line],["ff", "reglu", "geglu"])
plt.title("Test Accuracy vs Batch")
ax.set_xlabel("Batch")
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

'''
