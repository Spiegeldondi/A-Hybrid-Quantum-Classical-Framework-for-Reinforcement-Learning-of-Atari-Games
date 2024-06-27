import os
import csv
import numpy as np
import tensorflow as tf


class CustomLogger:
    def __init__(self):
        self.buffer = []

    def log(self, metrics):
        """
        Append the current metrics to the internal buffer

        metrics: list of evaluated metrics
        """
        self.buffer.append(metrics)

    def clear(self):
        self.buffer.clear()

    def write_to_csv(self, filepath, filename):
        # Check if directory exists and create if not
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        target = os.path.join(filepath, filename)
        with open(target, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.buffer)


def save_outputs(intermediate_models, observation, iteration, path="./"):

    save_dir = os.path.join(path, "layer_outputs")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    processed_observation = tf.convert_to_tensor(observation)
    processed_observation = tf.expand_dims(
        processed_observation, 0)  # Add batch dimension if needed

    for idx, inter_model in enumerate(intermediate_models):
        output = inter_model(processed_observation).numpy()
        filename = f"layer_{idx}_iter_{iteration}.npy"
        filepath = os.path.join(save_dir, f"layer_{idx}")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.save(os.path.join(filepath, filename), output)

    print(f"\nOutputs of iteration {iteration} stored in {save_dir}")


def save_weights(model, iteration, path="./"):

    save_dir = os.path.join(path, "layer_weights")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all layers weights for given iteration as npy file
    for idx, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:
            filename = f"layer_{idx}_iter_{iteration}.npy"
            filepath = os.path.join(save_dir, f"layer_{idx}")
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            np.save(os.path.join(filepath, filename), weights[0])

    print(f"\nWeights of iteration {iteration} stored in {save_dir}")


def save_biases(model, iteration, path="./"):

    save_dir = os.path.join(path, "layer_biases")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all layers biases for given iteration as npy file
    for idx, layer in enumerate(model.layers):
        biases = layer.get_weights()
        if biases:
            filename = f"layer_{idx}_iter_{iteration}.npy"
            filepath = os.path.join(save_dir, f"layer_{idx}")
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            try:
                np.save(os.path.join(filepath, filename), biases[1])
            except:
                pass

    print(f"\nBiases of iteration {iteration} stored in {save_dir}")


def save_weights_to_csv(model, layer_idx, save_dir="./weights_csv"):
    """
    Append the weights of a specific layer of the given model to a CSV file. 
    If the CSV does not exist, it will also save the dimensions of the weights.

    Parameters:
    - model: The TensorFlow or TF-Agents model whose weights need to be saved.
    - layer_idx: Index of the layer whose weights should be saved.
    - save_dir: The directory where the CSV files will be saved. Defaults to "./weights_csv".
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the specified layer
    layer = model.layers[layer_idx]

    # Check if the layer has weights
    if not layer.get_weights():
        print(f"Layer {layer_idx} does not have weights.")
        return

    # Retrieve the current weights of the layer and flatten them
    current_weights = layer.get_weights()[0]
    flattened_weights = current_weights.flatten()

    # Define the filename based on the layer's index
    filename = os.path.join(save_dir, f"layer_{layer_idx}_weights.csv")

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        # If the file is being created for the first time, write the dimensions
        if not os.path.exists(filename) or os.stat(filename).st_size == 0:
            writer.writerow(["DIMENSIONS"] + list(current_weights.shape))

        # Append the flattened weights
        writer.writerow(flattened_weights)

    print(f"\nWeights of layer {layer_idx} appended to CSV file in {save_dir}")


def save_weight_norms_to_csv(model, save_dir="./norms_csv"):
    """
    Append the norms of the weight matrices of a given model to a CSV file.

    Parameters:
    - model: The TensorFlow or TF-Agents model whose weight norms need to be saved.
    - save_dir: The directory where the CSV files will be saved. Defaults to "./norms_csv".
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the norms for all layers that have weights
    norms = []
    for layer in model.layers:
        if layer.get_weights():
            current_weights = layer.get_weights()[0]
            norm = np.linalg.norm(current_weights)
            norms.append(norm)

    # Define the filename
    filename = os.path.join(save_dir, "weight_norms.csv")

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        # Append the norms
        writer.writerow(norms)

    print(f"\nNorms of weights appended to CSV file in {save_dir}")


def create_intermediate_models(q_net):
    return [tf.keras.models.Sequential(q_net.layers[0:cutoff]) for cutoff in range(2, len(q_net.layers) + 1)]
