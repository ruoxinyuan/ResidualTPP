from collections import OrderedDict
import torch
import torch.nn.functional as F
import os
from easy_tpp.runner.base_runner import Runner
from easy_tpp.utils import RunnerPhase, logger, MetricsHelper, MetricsTracker, concat_element, save_pickle
from easy_tpp.utils.const import Backend
from easy_tpp.model.torch_model.torch_thinning import EventSampler


@Runner.register(name='std_tpp')
class TPPRunner(Runner):
    """Standard TPP runner
    """

    def __init__(self, runner_config, unique_model_dir=False, **kwargs):
        super(TPPRunner, self).__init__(runner_config, unique_model_dir, **kwargs)

        self.metrics_tracker = MetricsTracker()
        if self.runner_config.trainer_config.metrics is not None:
            self.metric_functions = self.runner_config.get_metric_functions()

        self._init_model()

        pretrain_dir = self.runner_config.model_config.pretrained_model_dir
        if pretrain_dir is not None:
            self._load_model(pretrain_dir)

    def _init_model(self):
        """Initialize the model.
        """
        self.use_torch = self.runner_config.base_config.backend == Backend.Torch

        if self.use_torch:
            from easy_tpp.utils import set_seed
            from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
            from easy_tpp.torch_wrapper import TorchModelWrapper
            from easy_tpp.utils import count_model_params
            set_seed(self.runner_config.trainer_config.seed)

            self.model = TorchBaseModel.generate_model_from_config(model_config=self.runner_config.model_config)
            self.model_wrapper = TorchModelWrapper(self.model,
                                                   self.runner_config.base_config,
                                                   self.runner_config.model_config,
                                                   self.runner_config.trainer_config)
            num_params = count_model_params(self.model)

        info_msg = f'Num of model parameters {num_params}'
        logger.info(info_msg)

    def _save_model(self, model_dir, **kwargs):
        """Save the model.

        Args:
            model_dir (str): the dir for model to save.
        """
        if model_dir is None:
            model_dir = self.runner_config.base_config.specs['saved_model_dir']
        self.model_wrapper.save(model_dir)
        logger.critical(f'Save model to {model_dir}')
        return

    def _load_model(self, model_dir, **kwargs):
        """Load the model from the dir.

        Args:
            model_dir (str): the dir for model to load.
        """
        self.model_wrapper.restore(model_dir)
        logger.critical(f'Load model from {model_dir}')
        return #self.model_wrapper.model

    def _train_model(self, train_loader, valid_loader, **kwargs):
        """Train the model.

        Args:
            train_loader (EasyTPP.DataLoader): data loader for the train set.
            valid_loader (EasyTPP.DataLoader): data loader for the valid set.
        """
        test_loader = kwargs.get('test_loader')
        for i in range(self.runner_config.trainer_config.max_epoch):
            train_metrics = self.run_one_epoch(train_loader, RunnerPhase.TRAIN)

            message = f"[ Epoch {i} (train) ]: train " + MetricsHelper.metrics_dict_to_str(train_metrics)
            logger.info(message)

            self.model_wrapper.write_summary(i, train_metrics, RunnerPhase.TRAIN)

            # evaluate model
            if i % self.runner_config.trainer_config.valid_freq == 0:
                valid_metrics = self.run_one_epoch(valid_loader, RunnerPhase.VALIDATE)

                self.model_wrapper.write_summary(i, valid_metrics, RunnerPhase.VALIDATE)

                message = f"[ Epoch {i} (valid) ]:  valid " + MetricsHelper.metrics_dict_to_str(valid_metrics)
                logger.info(message)

                updated = self.metrics_tracker.update_best("loglike", valid_metrics['loglike'], i)

                message_valid = "current best loglike on valid set is {:.4f} (updated at epoch-{})".format(
                    self.metrics_tracker.current_best['loglike'], self.metrics_tracker.episode_best)

                if updated:
                    message_valid += f", best updated at this epoch"
                    self.model_wrapper.save(self.runner_config.base_config.specs['saved_model_dir'])

                if test_loader is not None:
                    test_metrics = self.run_one_epoch(test_loader, RunnerPhase.VALIDATE)

                    message = f"[ Epoch {i} (test) ]: test " + MetricsHelper.metrics_dict_to_str(test_metrics)
                    logger.info(message)

                logger.critical(message_valid)
        
        self.model_wrapper.close_summary()

        load_dir = self.runner_config.base_config.specs['load_dir']
        if load_dir:
            self._compute_intensities()

        return
    
    def _compute_intensities(self):
        """Compute intensities for train and valid datasets using the trained model."""
        self.model_wrapper.model.eval()
        resweight = self.runner_config.data_config.data_specs.res_weight
        save_intensities_dir = self.runner_config.base_config.specs['saved_dir']
        load_dir = self.runner_config.base_config.specs['load_dir']
        accepted_dtimes_path = os.path.join(load_dir, 'accepted_dtimes.pth')
        if not os.path.exists(save_intensities_dir):
            os.makedirs(save_intensities_dir)
        with torch.no_grad():
            
            tensors = torch.load(os.path.join(load_dir, 'tensors.pth'))

            t_BN = tensors['time_seq']
            dt_BN = tensors['time_delta_seq']
            marks_BN = tensors['type_seq']
            seq_mask = tensors['seq_mask']
            sample_dt_BN = tensors['sample_time_delta_seq']
            bound_sample_dt_BN = tensors['dtime_for_bound_sampled']
            hawkes_bound_sample_intensities = tensors['bound_sampled_intensities']

            event_intensities = self.model_wrapper.model.compute_intensities_at_sample_times(
                time_seqs=t_BN[:, :-1], time_delta_seqs=dt_BN[:, :-1], type_seqs=marks_BN[:, :-1], sample_dtimes=dt_BN[:, 1:].unsqueeze(-1))
            event_intensities = event_intensities.squeeze(2)

            sample_intensities = self.model_wrapper.model.compute_intensities_at_sample_times(
                time_seqs=t_BN[:, :-1], time_delta_seqs=dt_BN[:, :-1], type_seqs=marks_BN[:, :-1], sample_dtimes=sample_dt_BN)

            intensities_for_bound = self.model_wrapper.model.compute_intensities_at_sample_times(
                time_seqs=t_BN[:, :-1], time_delta_seqs=dt_BN[:, :-1], type_seqs=marks_BN[:, :-1], sample_dtimes=bound_sample_dt_BN)

            intensities_for_bound = hawkes_bound_sample_intensities*(1-resweight) + intensities_for_bound*resweight
            intensity_upper_bound = intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * 5

            if os.path.exists(accepted_dtimes_path):
                accepted_dtimes = torch.load(accepted_dtimes_path)
                intensities_at_times = self.model_wrapper.model.compute_intensities_at_sample_times(
                    time_seqs=t_BN[:, :-1], time_delta_seqs=dt_BN[:, :-1], type_seqs=marks_BN[:, :-1], sample_dtimes=accepted_dtimes)
                
                torch.save(intensities_at_times, os.path.join(save_intensities_dir, 'intensities_at_times.pth'))
            else:
                print(f"File {accepted_dtimes_path} not found, skipping this part.")
            
        torch.save(event_intensities, os.path.join(save_intensities_dir, 'test_event_intensities.pth'))
        torch.save(sample_intensities, os.path.join(save_intensities_dir, 'test_sample_intensities.pth'))

        batch_size, seq_len = intensity_upper_bound.size()
        # num_exp = self.runner_config.model_config.thinning.get('num_exp', 500) 
        exp_numbers = torch.empty(size=[batch_size, seq_len, 50], dtype=torch.float32)
        exp_numbers.exponential_(1.0)
        exp_numbers = exp_numbers / intensity_upper_bound[:, :, None].float()
        exp_numbers = torch.cumsum(exp_numbers, dim=-1)  
        # print(intensity_upper_bound, exp_numbers)
        intensities_at_sampled_times = self.model_wrapper.model.compute_intensities_at_sample_times(
                    time_seqs=t_BN[:, :-1], time_delta_seqs=dt_BN[:, :-1], type_seqs=marks_BN[:, :-1], sample_dtimes=exp_numbers)
        torch.save({'intensities_at_sampled_times': intensities_at_sampled_times,
                    'intensity_upper_bound': intensity_upper_bound,
                    'exp_numbers': exp_numbers}, os.path.join(save_intensities_dir, 'thinning.pth'))
        
        logger.info("Computed and saved intensities for test datasets.")

        return 
    
    def _evaluate_model(self, data_loader, **kwargs):
        """Evaluate the model on the valid dataset.

        Args:
            data_loader (EasyTPP.DataLoader): data loader for the valid set

        Returns:
            dict: metrics dict.
        """

        eval_metrics = self.run_one_epoch(data_loader, RunnerPhase.VALIDATE)

        self.model_wrapper.write_summary(0, eval_metrics, RunnerPhase.VALIDATE)

        self.model_wrapper.close_summary()

        message = f"Evaluation result: " + MetricsHelper.metrics_dict_to_str(eval_metrics)

        logger.critical(message)

        return eval_metrics

    def _gen_model(self, data_loader, **kwargs):
        """Generation of the TPP, one-step and multi-step are both supported.
        """

        test_result = self.run_one_epoch(data_loader, RunnerPhase.PREDICT)

        # For the moment we save it to a pkl

        message = f'Save the prediction to pickle file pred.pkl'

        logger.critical(message)

        save_pickle('pred.pkl', test_result)

        return

    def run_one_epoch(self, data_loader, phase):
        """Run one complete epoch.

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]

        Returns:
            a dict of metrics
        """
        total_loss = 0
        total_num_event = 0
        epoch_label = []
        epoch_pred = []
        epoch_mask = []
        pad_index = self.runner_config.data_config.data_specs.pad_token_id
        metrics_dict = OrderedDict()
        if phase in [RunnerPhase.TRAIN, RunnerPhase.VALIDATE]:
            for batch in data_loader:
                batch_loss, batch_num_event, batch_pred, batch_label, batch_mask = \
                    self.model_wrapper.run_batch(batch, phase=phase)

                total_loss += batch_loss
                total_num_event += batch_num_event
                epoch_pred.append(batch_pred)
                epoch_label.append(batch_label)
                epoch_mask.append(batch_mask)

            avg_loss = total_loss / total_num_event

            metrics_dict.update({'loglike': -avg_loss, 'num_events': total_num_event})

        else:
            for batch in data_loader:
                batch_pred, batch_label = self.model_wrapper.run_batch(batch, phase=phase)
                epoch_pred.append(batch_pred)
                epoch_label.append(batch_label)

        # we need to improve the code here
        # classify batch_output to list
        pred_exists, label_exists = False, False
        if epoch_pred[0][0] is not None:
            epoch_pred = concat_element(epoch_pred, pad_index)
            pred_exists = True
        if len(epoch_label) > 0 and epoch_label[0][0] is not None:
            epoch_label = concat_element(epoch_label, pad_index)
            label_exists = True
            if len(epoch_mask):
                epoch_mask = concat_element(epoch_mask, False)[0]  # retrieve the first element of concat array
                epoch_mask = epoch_mask.astype(bool)

        if pred_exists and label_exists:
            metrics_dict.update(self.metric_functions(epoch_pred, epoch_label, seq_mask=epoch_mask))

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({'pred': epoch_pred, 'label': epoch_label})

        return metrics_dict
