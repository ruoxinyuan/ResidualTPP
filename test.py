import argparse
import torch
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
from easy_tpp.preprocess import TPPDataLoader

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/origin_taxi.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='RMTPP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
    model_runner = Runner.build_from_config(config)

    model_runner.run()
    # model_runner.save('best_models/NHP_model.pth')

    # best_model = model_runner.load('best_models/NHP_model.pth')
    # best_model.eval()

    # # 从配置文件中读取 batch_size 和 shuffle 参数
    # batch_size = config.trainer_config.batch_size
    # shuffle = config.trainer_config.shuffle

    # # 准备数据
    # data_loader = TPPDataLoader(config.data_config, batch_size=batch_size, shuffle=shuffle)
    # train_loader = data_loader.train_loader()

    # # 计算所有训练集数据上的 intensities
    # all_intensities = []
    # with torch.no_grad():
    #     for batch in train_loader:
    #         t_BN, dt_BN, marks_BN, _, _ = batch
    #         print(t_BN, dt_BN, marks_BN)
    #         intensities = best_model.compute_intensities_at_sample_times(time_seqs=t_BN, time_delta_seqs=dt_BN, type_seqs=marks_BN, sample_dtimes=dt_BN)
    #         all_intensities.append(intensities)

    # # 将所有 intensities 合并
    # all_intensities = torch.cat(all_intensities, dim=0)
    # print(all_intensities)


if __name__ == '__main__':
    main()