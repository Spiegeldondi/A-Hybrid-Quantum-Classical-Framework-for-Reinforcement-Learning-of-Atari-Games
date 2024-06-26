"""

---> COPYRIGHT DISCLAIMER HERE <---

"""

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
            