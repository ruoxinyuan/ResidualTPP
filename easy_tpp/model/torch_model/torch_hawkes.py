import torch
from torch import nn
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

class Hawkes(TorchBaseModel):
    def __init__(self, model_config):
        super(Hawkes, self).__init__(model_config)
        self.alpha = nn.Parameter(torch.Tensor(self.num_event_types, self.num_event_types))
        self.beta = nn.Parameter(torch.Tensor(self.num_event_types, self.num_event_types))
        self.mu = nn.Parameter(torch.Tensor(self.num_event_types))
        nn.init.xavier_uniform_(self.alpha)
        nn.init.xavier_uniform_(self.beta)
        nn.init.uniform_(self.mu)

    def compute_intensities_at_sample_times(self, history, current_time, batch_non_pad_mask):
        """
        Compute the intensity function for the Hawkes process, considering padding mask.

        Args:
            history (tensor): [batch_size, seq_len, num_event_types], history of events.
            current_time (tensor): [batch_size, seq_len, num_sample], current time for each event.
            batch_non_pad_mask (tensor): [batch_size, seq_len], mask for non-padding entries.

        Returns:
            tensor: [batch_size, seq_len, num_sample, num_event_types], intensity for each event type.
        """
        # Compute the decay term (time difference between events)
        time_diff = torch.clamp(current_time.unsqueeze(-1) - history.unsqueeze(2), min=0.0)
        decay = torch.exp(-self.beta * time_diff)

        # We apply the non-padding mask (batch_non_pad_mask) to ignore padding events in the calculation.
        decay = decay * batch_non_pad_mask.unsqueeze(-1).unsqueeze(-1)  # Broadcasting mask to decay tensor

        # Compute the intensity using alpha and decay
        intensity = self.mu + torch.sum(self.alpha * decay, dim=1)  # Sum over history to get intensity

        # We apply the non-padding mask again to ensure that we ignore padding in the final output
        intensity = intensity * batch_non_pad_mask  # This ensures intensity for padding is zero

        return intensity
    

    def forward(self, time_seqs, batch_non_pad_mask):
        """
        Forward pass for the Hawkes process model with batch input.

        Returns:
            tensor: [batch_size, seq_len, num_event_types], intensity for each event type.
        """
        
        # Ensure the batch size and sequence length
        batch_size, seq_len = time_seqs.size()

        # Construct the history and current_time tensors
        history = time_seqs.unsqueeze(1).expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
        current_time = time_seqs.unsqueeze(-1).expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

        # Compute intensity considering padding
        intensity = self.compute_intensities_at_sample_times(history, current_time, batch_non_pad_mask).squeeze(-2)

        return intensity
    
    
    # Compute the loglikelihood loss
    def loglike_loss(self, batch):
        """Compute the log-likelihood loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # Compute the intensity at event times
        lambda_at_event = self.forward(ts_BN[:, :-1], batch_non_pad_mask[:, :-1])

        # Sample times for non-event log-likelihood computation
        sample_dtimes = self.make_dtime_loss_samples(dts_BN[:, 1:])

        # Compute the intensity at sampled times
        lambda_t_sample = self.compute_intensities_at_sample_times(ts_BN, sample_dtimes, batch_non_pad_mask)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=dts_BN[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:]
        )

        # Compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events


    # Compute the intensities at given sampling times
    # Used in the Thinning sampler
    # def compute_intensities_at_sample_times(self, batch, sample_times, **kwargs):
    #     ...
    #     return intensities