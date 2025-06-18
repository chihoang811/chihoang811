import numpy as np
import os

def generate_samples(N, num_points=100, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 2 * np.pi, num_points)
    inputs = []
    targets = []

    for _ in range(N):
        p_sin = np.random.uniform(-1, 1)
        p_cos = np.random.uniform(-1, 1)

        u = p_sin * np.sin(x) + p_cos * np.cos(x)
        v = p_sin * np.cos(x) - p_cos * np.sin(x)  # derivative

        inputs.append(u)
        targets.append(v)

    return x, np.array(inputs), np.array(targets)

def save_data(path, x, u, v):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "x.npy"), x)
    np.save(os.path.join(path, "u.npy"), u)
    np.save(os.path.join(path, "v.npy"), v)

if __name__ == "__main__":
    # Parameters
    num_train = 10000
    num_test = 1000
    num_points = 100

    # Generate
    x, u_train, v_train = generate_samples(num_train, num_points)
    _, u_test, v_test = generate_samples(num_test, num_points, seed=999)

    # Save
    save_data("data/train", x, u_train, v_train)
    save_data("data/test", x, u_test, v_test)

    print("Data generated and saved in 'data/train' and 'data/test'.")
