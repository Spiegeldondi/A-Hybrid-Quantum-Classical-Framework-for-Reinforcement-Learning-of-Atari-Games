import os
import argparse

def is_valid_id(s):
    if s.isdigit() and len(s) == 2:
        return s
    else:
        raise argparse.ArgumentTypeError("ID must be a 2-digit number.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model for the Atari Breakout game.')
    parser.add_argument('model', type=str, choices=['classic', 'quantum'], help='Specify the model type: "classic" or "quantum".')
    parser.add_argument('activation', type=str, choices=['linear', 'tanh', 'relu'], help='Specify the activation function before the PQC')
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
    parser.add_argument("--path", type=str, required=True, help="provide a path for output files")

    args = parser.parse_args()

    if args.model == 'quantum' and args.lr6 is None:
        parser.error('Model type "quantum" requires 6 learning rates.')
    elif args.model == 'classic' and args.lr6 is not None:
        parser.error('Model type "classic" requires 5 learning rates and no 6th learning rate should be provided.')

    if args.model == 'classic' and args.bottleneck is None:
        parser.error('Please specify if bottleneck layer should be inserted')
    elif args.model == 'quantum' and args.bottleneck is not None:
        parser.error('Bottleneck layer not sensible in quantum model')

    return args
