import argparse
import config 
import utils
import quantum 
import agents

def main():
    args = config.parse_arguments()

    print(f"\nRun ID: {args.id} \nRunning {args.model} model with {args.n_layers} layers, " + 
          f"{args.activation} activation and learning rates: " +
          f"\n{args.lr1}, {args.lr2}, {args.lr3}, {args.lr4}, {args.lr5}, {args.lr6} " + 
          f"\nSaving output to {args.path}\n")



if __name__ == "__main__":
    main()