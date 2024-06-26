import argparse
import config 
import setup
# import utils
# import quantum 
# import agents

def main():
    args = config.parse_arguments()

    print(f"\nRun ID: {args.id} \nRunning {args.model} model with {args.n_layers} layers, " + 
          f"{args.activation} activation and learning rates: " +
          f"\n{args.lr1}, {args.lr2}, {args.lr3}, {args.lr4}, {args.lr5}, {args.lr6} " + 
          f"\nSaving output to {args.path}\n")
    
    # Seeds must be given to the agents module somehow but also to the quantum module because of the angle initialization
    # tf.random.set_seed(rnd_seed)
    # np.random.seed(rnd_seed)
    
    LOGGING_INTERVAL = 1000
    CHECKPOINT_INTERVAL = 1000

    log_dir = setup.create_log_dir(args.path, args.id)
    log_files = setup.create_log_files(log_dir)
    
    # # If no log files are found start training
    # if len(log_files) == 0:
    #     print("\nNo previous log files found. Starting a fresh run.\n")
    #     agents.train_agent(n_iterations=2500000)

    # # If one log file is found resume training
    # elif len(log_files) == 1:
    #     print(f"\nLog file for run {args.id} found. Continuing training.\n")
    #     # Restore checkpoint
    #     agents.train_checkpointer.initialize_or_restore()
    #     agents.train_agent(n_iterations=2500000)

    # # If multiple log files are found, throw an exception
    # else:
    #     raise Exception("\nMultiple log files found. Unclear which one to use.\n")



if __name__ == "__main__":
    main()
    