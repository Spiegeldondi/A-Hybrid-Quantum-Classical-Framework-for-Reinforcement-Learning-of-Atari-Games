import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import csv
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import gym

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

print(tf.config.list_physical_devices())

from tf_agents.environments import suite_atari, suite_gym
# from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import sequential
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from tf_agents.utils.common import Checkpointer
from tf_agents.eval.metric_utils import MetricsGroup



def is_valid_id(s):
    if s.isdigit() and len(s) == 2:
        return s
    else:
        raise argparse.ArgumentTypeError("ID must be a 2-digit number.")

# Create argument parser
parser = argparse.ArgumentParser(description='Train a model for the Atari Breakout game.')

# Add parser arguments
parser.add_argument('model', type=str, choices=['classic', 'quantum'],
                    help='Specify the model type: "classic" or "quantum".')
parser.add_argument('activation', type=str, choices=['linear', 'tanh', 'relu'],
                    help='Specify the activation function before the PQC')
parser.add_argument('lr1', type=float, help='Learning rate for the 1st layer.')
parser.add_argument('lr2', type=float, help='Learning rate for the 2nd layer.')
parser.add_argument('lr3', type=float, help='Learning rate for the 3rd layer.')
parser.add_argument('lr4', type=float, help='Learning rate for the 4th layer.')
parser.add_argument('lr5', type=float, help='Learning rate for the 5th layer.')
parser.add_argument('--lr6', type=float, help='Learning rate for the 6th layer.')
parser.add_argument("--n_qubits", type=int, required=True, help="number of qubits in the PQC")
parser.add_argument("--n_layers", type=int, required=True, help="number of layers in the PQC")
parser.add_argument("--scaling", type=int, required=True, help="scaling applied to rewards")
parser.add_argument('--bottleneck', type=int, choices=[0, 1], help='Specify if bottleneck layer.')
parser.add_argument("--id", type=is_valid_id, required=True, help="2-digit unique ID")
parser.add_argument("--seed", type=int, required=True, help="random seed")
parser.add_argument("--path", type=str, help="provide a path for output files")

args = parser.parse_args()

# Validate the learning rates based on the model type
if args.model == 'quantum' and args.lr6 is None:
    parser.error('Model type "quantum" requires 6 learning rates.')
elif args.model == 'classic' and args.lr6 is not None:
    parser.error('Model type "classic" requires 5 learning rates and no 6th learning rate should be provided.')

if args.model == 'classic' and args.bottleneck is None:
    parser.error('Please specify if bottleneck layer should be inserted')
elif args.model == 'quantum' and args.bottleneck is not None:
    parser.error('Bottleneck layer not sensible in quantum model')

output_path = args.path
rnd_seed = args.seed
run_id = args.id 
model = args.model
bottleneck = args.bottleneck
activation = args.activation
lr1 = args.lr1
lr2 = args.lr2
lr3 = args.lr3
lr4 = args.lr4
lr5 = args.lr5
lr6 = args.lr6  # This will be None for classic model
n_layers = args.n_layers
n_qubits = args.n_qubits
scaling = args.scaling

print(f"\nRun ID: {run_id} \nRunning {model} model with {n_layers} layers, {activation} activation and learning rates: " +
      f"\n{lr1}, {lr2}, {lr3}, {lr4}, {lr5}, {lr6} " + 
      f"\nSaving output to {output_path}")

tf.random.set_seed(rnd_seed)
np.random.seed(rnd_seed)

logging_interval = 1000
checkpoint_interval = 1000

if output_path is None:
    log_dir = os.path.join(os.getcwd(), "output-"+args.id)
else:
    log_dir = os.path.join(output_path, "output-"+args.id)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_files = [f for f in os.listdir(log_dir) if f.startswith("metrics") and f.endswith('.csv')]
print("\nlen(log_files):\n", len(log_files))

# ### Parametrized Quantum Circuit

# In[ ]:


import cirq
import sympy
import tensorflow_quantum as tfq


# In[ ]:


def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


# In[ ]:


def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)


# In[ ]:


class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)
        
        q_delta = 0.01
        theta_init = tf.random_normal_initializer(mean=0.0, stddev=q_delta*np.pi, seed=rnd_seed)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs, multiples=[1, self.n_layers])

        joined_vars = tf.concat([tiled_up_thetas, tiled_up_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])



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
    processed_observation = tf.expand_dims(processed_observation, 0)  # Add batch dimension if needed

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


def create_intermediate_models(model):
    return [tf.keras.models.Sequential(q_net.layers[0:cutoff]) for cutoff in range(2, len(q_net.layers) + 1)]
# ### TF-Agents Environments

# In[ ]:

class ReducedActionSpace(gym.Wrapper):
    def __init__(self, environment, scale=10):
        super(ReducedActionSpace, self).__init__(environment)
        # Including NOOP (0), RIGHT (2), and LEFT (3) in the action map
        self._action_map = [0, 2, 3]
        self.action_space = gym.spaces.Discrete(len(self._action_map))
        self.scale = scale  # Store the scale

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(1) # FIRE to start
        return obs

    def step(self, action):
        lives_before_action = self.ale.lives()
        # Map the action using the reduced action space
        mapped_action = self._action_map[action]
        obs, rewards, done, info = super().step(mapped_action)
        rewards *= self.scale  # Scale the rewards
        # Check if a life was lost and the game is not done, then FIRE
        if self.ale.lives() < lives_before_action and not done:
            super().step(1)  # FIRE to start after life lost
        return obs, rewards, done, info

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"

env = suite_gym.load(environment_name,
                     gym_env_wrappers=[lambda env: gym.wrappers.AtariPreprocessing(env, 
                                                                                   grayscale_newaxis=True, 
                                                                                   noop_max=0),
                                       lambda env: ReducedActionSpace(env, scale=scaling),
                                       FrameStack4])

# In[ ]:


env.seed(rnd_seed)


# In[ ]:


# Convert the Python environment to a TF environment
tf_env = TFPyEnvironment(env)

print("\n", tf_env.action_spec(), "\n")


# ### Creating the DQN

if model == "quantum":
    print("\nRunning quantum model\n")
    qubits = cirq.GridQubit.rect(1, n_qubits)   
    observables = [cirq.Z(q) for q in qubits]
    
    preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)

    # Convolutions on the frames on the screen
    layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")
    layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")
    layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")

    layer4 = keras.layers.Flatten()

    layer5 = keras.layers.Dense(n_qubits * n_layers, activation=activation)
    layer6 = ReUploadingPQC(qubits, n_layers, observables, name="pqc")
    action = keras.layers.Dense(3, activation="linear")
    # action = Rescaling(input_dim=num_actions)

    q_net = sequential.Sequential([preprocessing_layer, layer1, layer2, layer3, layer4, layer5, layer6, action])
    
    # Set the learning rate for each layer
    learning_rates = [lr1, lr1, # layer 1
                      lr2, lr2, # layer 2
                      lr3, lr3, # layer 3
                      lr4, lr4, # layer 5
                      lr5, # layer 6
                      lr6, lr6, # action
                     ]

    optimizer = [
        keras.optimizers.Adam(learning_rate=lr) for lr in learning_rates
    ]
    
elif model == "classic":
    print("\nRunning classic model\n")
    preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)

    # Convolutions on the frames on the screen
    layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")
    layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")
    layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")

    layer4 = keras.layers.Flatten()
    bottleneck_layer = keras.layers.Dense(n_qubits * n_layers, activation=activation)
    layer5 = keras.layers.Dense(512, activation="relu")
    action = keras.layers.Dense(3, activation="linear")

    if bottleneck:
        q_net = sequential.Sequential([preprocessing_layer, layer1, layer2, layer3, layer4, bottleneck_layer, layer5, action])

        # Set the learning rate for each layer
        learning_rates = [lr1, lr1, # layer 1
                        lr2, lr2, # layer 2
                        lr3, lr3, # layer 3
                        lr4, lr4,  # bottleneck
                        lr4, lr4, # layer 5
                        lr5, lr5, # action
                        ]
    elif not bottleneck:
        q_net = sequential.Sequential([preprocessing_layer, layer1, layer2, layer3, layer4, layer5, action])

        # Set the learning rate for each layer
        learning_rates = [lr1, lr1, # layer 1
                        lr2, lr2, # layer 2
                        lr3, lr3, # layer 3
                        lr4, lr4, # layer 5
                        lr5, lr5, # action
                        ]

    optimizer = [
        keras.optimizers.Adam(learning_rate=lr) for lr in learning_rates
    ]

else:
    raise ValueError(f"Invalid model: {model}. Expected 'quantum' or 'classic'.")


# In[ ]:

from tf_agents.utils import common

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps

epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε

loss_fn = common.element_wise_squared_loss

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=loss_fn,
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))

agent.initialize()


# In[ ]:


if model == "quantum":
    print(f"\n{len(optimizer)} Optimizers are in use.\n")
    print(f"\n{len(q_net.trainable_variables)} sets of trainable variables.\n")


# In[ ]:


print("\n", q_net.summary(), "\n")


# In[ ]:


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000) # reduce if OOM error


replay_buffer_observer = replay_buffer.add_batch


# In[ ]:


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


# In[ ]:


# Add some training metrics:
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(buffer_size=1),
    tf_metrics.AverageReturnMetric(buffer_size=10),
    tf_metrics.AverageReturnMetric(buffer_size=100),
    tf_metrics.AverageReturnMetric(buffer_size=1000),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=1),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=10),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=100),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=1000),
    # tf_metrics.ChosenActionHistogram(), throws a ValueError int32 vs. int64
    tf_metrics.MinReturnMetric(),
    tf_metrics.MaxReturnMetric(),
]


# In[ ]:


# Create the collect driver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration


# In[ ]:


# Collect the initial experiences, before training
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000 ALE frames

# If no log files are found, collect initial experiences
if len(log_files) == 0:
    final_time_step, final_policy_state = init_driver.run()


# In[ ]:


# Create the dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)


# In[ ]:


# Convert the main functions to TF Functions for better performance
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


# In[ ]:


# Setup up custom logger
custom_logger = CustomLogger()
reward_logger = CustomLogger()
is_last_logger = CustomLogger()


# In[ ]:


# Setup Checkpointer 
checkpoint_dir = os.path.join(log_dir, 'checkpoint')

train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    train_step=train_step,
    metrics=MetricsGroup(train_metrics, 'train_metrics')
)


# In[ ]:

intermediate_models = create_intermediate_models(q_net)


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    
    # Start timer
    start = time.time()
    
    # Count iterations
    iteration = train_step.numpy()

    while iteration < n_iterations:
        time_step, policy_state = collect_driver.run(time_step, policy_state)

        # Check if the episode has ended
        if time_step.is_last().numpy()[0]:
            # Get the current observation
            processed_observation = tf.convert_to_tensor(time_step.observation)
            processed_observation = tf.expand_dims(processed_observation, 0)  # Add batch dimension if needed

            # Get Q-values from the Q-network
            q_values = q_net(processed_observation)[0].numpy()  

            is_last_logger.log(q_values[0]) 

        # Log the reward and discount for every timestep
        reward_logger.log([time_step.reward.numpy()[0], 
                           time_step.discount.numpy()[0], 
                           time_step.step_type.numpy()[0]])

	    # Save weights and outputs       
        if iteration % 10000 == 0:
            save_weights(q_net, iteration, path=log_dir)
            save_biases(q_net, iteration, path=log_dir)
            save_outputs(intermediate_models, time_step.observation, iteration, log_dir)

        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
         
        # Update train_step counter
        iteration = train_step.numpy()
        
        # Print status to console
        print("\r{} loss: {:.5f}".format(
            train_step.numpy(), train_loss.loss.numpy()), end="")

        if iteration % logging_interval == 0:
            end = time.time()
            
            # Log training metrics
            log_metrics(train_metrics)

            # Log training metrics
            custom_logger.log([agent.train_step_counter.numpy()] + 
                              [m.result().numpy() for m in train_metrics] + 
                              [round(end-start)])
            
            # Write logged metrics to file
            custom_logger.write_to_csv(log_dir, "metrics.csv")
            custom_logger.clear()

            reward_logger.write_to_csv(log_dir, "rewards.csv")
            reward_logger.clear()

            is_last_logger.write_to_csv(log_dir, "is_last.csv")
            is_last_logger.clear()
                
            # Save norm of weights matrices of all layers
            save_weight_norms_to_csv(q_net, save_dir=log_dir)

            # Reset time
            start = time.time()
        
        if iteration % checkpoint_interval == 0:
            # Save to checkpoint
            train_checkpointer.save(train_step)


# In[ ]:


# If no log files are found start training
if len(log_files) == 0:
    print("\nNo previous log files found. Starting a fresh run.\n")
    train_agent(n_iterations=2500000)

# If one log file is found resume training
elif len(log_files) == 1:
    print(f"\nLog file for run {run_id} found. Continuing training.\n")
    # Restore checkpoint
    train_checkpointer.initialize_or_restore()
    train_agent(n_iterations=2500000)

# If multiple log files are found, throw an exception
else:
    raise Exception("\nMultiple log files found. Unclear which one to use.\n")
