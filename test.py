import argparse
import torch
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
from easy_tpp.preprocess import TPPDataLoader

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='example/configs/origin_earthquake.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='RMTPP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
    main()