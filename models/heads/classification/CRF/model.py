from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers.activations import ACT2FN

from models.ops import logsumexp

from models.heads.classification.CRF.periprocess import viterbi_decode


class ConditionalRandomField(nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute log-likelihood of its inputs 
        assuming a conditional random field model.
    
    Reference: 
        http://www.cs.columbia.edu/~mcollins/fb.pdf
    
    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    constraints : List[Tuple[int, int]], optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_transitions : bool, optional (default: True)
        Whether to include the start and end transition parameters.
    """
    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]] = None,
                                                include_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence      (num_tags + 1) 
        #               and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags+2, num_tags+2).fill_(1.)
        else:
            constraint_mask = torch.Tensor(num_tags+2, num_tags+2).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_transitions = include_transitions
        if include_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, 
        which is the sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods, 
        # combining the transitions to the initial states and the logits for the first timestep.
        if self.include_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are (instance, current_tag, next_tag)
        for i in range(1, sequence_length):

            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)

            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)

            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            alpha = logsumexp(inner, 1) *      mask[i].view(batch_size, 1) + \
                                  alpha * (1 - mask[i]).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape
        mask = mask.float()

        # Transpose batch size and sequence dimensions:
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition_score if next element is unmasked,
            #              input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. 
        # To start with, we need to find the last tag for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.ByteTensor = None, 
                      input_batch_first: bool=False, keepdim: bool=False) -> torch.Tensor:
        """
        Computes the log likelihood. inputs, tags, mask are assumed to be batch first
        """
        # convert to batch_first
        if not input_batch_first:
            if mask is not None:
                mask = mask.transpose(0, 1).contiguous()
            inputs = inputs.transpose(0, 1).contiguous()
            tags = tags.transpose(0, 1).contiguous()

        # pylint: disable=arguments-differ
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        if keepdim:
            return log_numerator - log_denominator
        else:
            return (log_numerator - log_denominator).sum()

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.Tensor, 
                           logits_batch_first: bool=False) -> List[Tuple[List[int], float]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        """
        if not logits_batch_first:
            logits = logits.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits = logits.data
        mask = mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags+2, num_tags+2).fill_(-10_000.)

        # Apply transition constraints
        constrained_transitions = self.transitions *      self._constraint_mask[:num_tags, :num_tags] + \
                                         -10_000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions.detach() *      self._constraint_mask[start_tag, :num_tags].data + \
                                                                      -10_000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] =     self.end_transitions.detach() *      self._constraint_mask[:num_tags,  end_tag].data + \
                                                                      -10_000.0 * (1 - self._constraint_mask[:num_tags,  end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10_000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags,  end_tag ] = -10_000.0 * (1 - self._constraint_mask[ :num_tags, end_tag ].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length+2, num_tags+2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10_000.)

            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, viterbi_score = viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))

        return best_paths


