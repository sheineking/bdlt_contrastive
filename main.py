import argparse
import json
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='Paraphrase classifier')

    # First, define the mode (contrastive, supervised or pretrained) and base configuration to be used
    parser.add_argument('--mode', metavar='mode', type=str, required=True,
                        help='"contrastive", "supervised" or "pretrained". Chooses which kind of training to conduct.')
    parser.add_argument('--config', metavar='config', type=str, required=True,
                        help='Name of the base configuration. Used as key for the dictionary saved '
                             'in model_configs.json')

    # Optionally, define parameters to be changed in the configuration
    # Learning Manager parameters
    parser.add_argument('--model_name', metavar='model_name', type=str, required=False,
                        help='Name used to identify the model weights and logs')
    parser.add_argument('--encoder', metavar='encoder', type=str, required=False,
                        help='For pretrained models: Name of a contrastive model to be used as encoder.')

    # Training parameters
    parser.add_argument('--epochs', metavar='epochs', type=int, required=False,
                        help='How many epochs to train the model')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, required=False,
                        help='Size of one batch used in training')
    parser.add_argument('--optimizer_name', metavar='optimizer_name', type=str, required=False,
                        help='Name of the optimizer to be used (sdg, rmsprop, lars)')
    parser.add_argument('--lr', metavar='lr', type=float, required=False,
                        help='Learning rate used by the optimizer')
    parser.add_argument('--weight_decay', metavar='weight_decay', type=float, required=False,
                        help='Weight Decay for SGD or RMSProp')
    parser.add_argument('--alpha', metavar='alpha', type=float, required=False,
                        help='Alpha for RMSProp')
    parser.add_argument('--eps', metavar='eps', type=float, required=False,
                        help='Epsilon for RMSProp or LARS')
    parser.add_argument('--trust_coef', metavar='trust_coef', type=float, required=False,
                        help='Trust coefficient for LARS')

    return parser.parse_args()



def update_params(args, configs):
    # Get the param_dict based on the config argument
    param_dict = configs[args.config]

    # Update Learning Manager parameters
    param_dict['learning_manager']['model_name'] = args.model_name if args.model_name is not None \
        else param_dict['learning_manager']['model_name']
    param_dict['learning_manager']['encoder'] = args.encoder if args.encoder is not None \
        else param_dict['learning_manager']['encoder']

    # Update Training parameters
    param_dict['training']['epochs'] = args.epochs if args.epochs is not None \
        else param_dict['training']['epochs']
    param_dict['training']['batch_size'] = args.batch_size if args.batch_size is not None \
        else param_dict['training']['batch_size']
    param_dict['training']['optimizer_name'] = args.optimizer_name if args.optimizer_name is not None \
        else param_dict['training']['optimizer_name']
    param_dict['training']['lr'] = args.lr if args.lr is not None \
        else param_dict['training']['lr']
    param_dict['training']['weight_decay'] = args.weight_decay if args.weight_decay is not None \
        else param_dict['training']['weight_decay']
    param_dict['training']['alpha'] = args.alpha if args.alpha is not None \
        else param_dict['training']['alpha']
    param_dict['training']['eps'] = args.eps if args.eps is not None \
        else param_dict['training']['eps']
    param_dict['training']['trust_coef'] = args.trust_coef if args.trust_coef is not None \
        else param_dict['training']['trust_coef']

    return param_dict

def switch_directory(folder_name='./contrastive/'):
    abs_path = os.path.abspath(folder_name + "learning_manager.py")
    dir_name = os.path.dirname(abs_path)

    sys.path.append(dir_name)
    os.chdir(dir_name)


if __name__ == "__main__":
    # Argument Parsing
    args = parse_arguments()

    # Switch directory to import the correct modules
    if args.mode == "supervised" or args.mode == "pretrained":
        switch_directory('./supervised/')

    elif args.mode == "contrastive":
        switch_directory('./contrastive/')

    else:
        print("Please provide a valid mode: 'contrastive', 'supervised' or 'pretrained' ")
        exit(1)

    import learning_manager as l
    import model_configs as c

    # Parameter Updating
    configs = json.load(open(c.CONFIG_PATH))
    param_dict = update_params(args, configs)

    # Apply the param_dict
    Manager = l.LearningManager(**param_dict['learning_manager'])
    Manager.conduct_training(**param_dict['training'])
