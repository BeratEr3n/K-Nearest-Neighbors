import random
import math
import matplotlib.pyplot as plt

def create_data(how_many):
    data = []
    for i in range(0, int(how_many)):
        if i % 3 == 0:
            data.append({"x": random.randint(0, 1000), "y": random.randint(0, 1000), "label": "A"})
        elif i % 3 == 1:
            data.append({"x": random.randint(0, 1000), "y": random.randint(0, 1000), "label": "B"})
        else:
            data.append({"x": random.randint(0, 1000), "y": random.randint(0, 1000), "label": "C"})
    return data

def euclidean_distance(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

def normalize_data(data, max_x, max_y):
    for d in data:
        d["x"] /= max_x
        d["y"] /= max_y
    return data

def KNN(data, point, k):
    distances = [(d, euclidean_distance(point, d)) for d in data]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    k_nearest = sorted_distances[:k]
    labels = [d[0]["label"] for d in k_nearest]
    return max(set(labels), key=labels.count)

test_point = {"x": int(input("Enter test point x value:")), "y": int(input("Enter test point y value:"))}
k = int(input("Enter k value:"))
how_many = int(input("How many data points:"))

data = create_data(how_many)
max_x = max(d["x"] for d in data)
max_y = max(d["y"] for d in data)
data = normalize_data(data, max_x, max_y)

test_point = {"x": 500, "y": 500}
test_point["x"] /= max_x
test_point["y"] /= max_y

for d in data:
    color = 'r' if d["label"] == "A" else ('g' if d["label"] == "B" else 'b')
    marker = 'o' if d["label"] == "A" else ('s' if d["label"] == "B" else '^')
    plt.scatter(d["x"], d["y"], color=color, marker=marker)

plt.scatter(test_point["x"], test_point["y"], color='k', marker='x', label='Test Point')

prediction = KNN(data, test_point, k)
plt.title(f"KNN Prediction: {prediction}")
plt.legend()
plt.show()
