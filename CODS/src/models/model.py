
'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
from json import encoder
import os
import logging
import inspect
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BertModel,
    LogitsProcessorList,
    StoppingCriteriaList
)
from transformers.generation_utils import (
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput
)
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.models.bart.modeling_bart import _expand_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.generation_beam_search import BeamSearchScorer

from src.utils.constants import DEVICE

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]

logger = logging.getLogger(__name__)

global inf
inf = 1e10


class TypedModel(nn.Module):
    """
    Two types of dialogs: useful information(marked as 'function_dialogs' in data) / useless chitchats
    Predict these two types
    """

    def __init__(self, bert_path='bert-base-uncased', num_labels=2):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, evaluate=False):
        document = data['document'].to(DEVICE)
        label = data['label'].to(DEVICE)

        output = self.model(document)
        cls_vector = output[1]

        prediction_logits = self.classifier(cls_vector)
        loss = F.cross_entropy(prediction_logits, label)
        if evaluate:
            return loss, prediction_logits
        else:
            return loss


class SegmentPredictor(nn.Module):
    def __init__(self, bert_path='bert-base-uncased', num_labels=2):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.num_labels = num_labels

    @property
    def device(self):
        return next(self.parameters()).device

    def get_loss(self, data, person, document, turn_nums):
        segment_label = data[f'segment_label_{person}'].to(DEVICE)
        output = self.model(document)
        all_seq_hs = output[0]  # batch_size, seq_len, hd_dim

        sent_repr_mat = []
        max_turn_num = max(turn_nums)
        for i in range(all_seq_hs.size(0)):
            sent_repr = all_seq_hs[i][document[i] == 101]  # [num_of_turns, hd_dim]
            sent_repr = torch.cat(
                [sent_repr, torch.zeros(max_turn_num - turn_nums[i], sent_repr.size(1)).to(DEVICE)], 0)
            sent_repr_mat.append(sent_repr)
            segment_label[i][turn_nums[i]:] = -1
        sent_repr_mat = torch.stack(sent_repr_mat, 0)  # [batch_size, max_turn_num, hd_dim]
        segment_label = segment_label[:, :max_turn_num]
        prediction_logits = self.classifier(sent_repr_mat)
        loss = F.cross_entropy(prediction_logits.reshape(-1, self.num_labels), segment_label.reshape(-1), ignore_index=-1,
                               reduction='mean')
        return loss, prediction_logits, segment_label

    def prepare_eval(self, prediction_logits, turn_nums):
        batch_size = prediction_logits.size(0)
        eval_prediction_logits = []
        for i in range(batch_size):
            eval_prediction_logits.append(prediction_logits[i][:turn_nums[i], :])
        
        return eval_prediction_logits

    def forward(self, data, evaluate=False):
        document = data['document'].to(DEVICE)
        turn_nums = [(item == 101).sum().cpu().item() for item in document]
        loss_1, pred_logits_1, segment_label_1 = self.get_loss(data, '1', document, turn_nums)
        loss_2, pred_logits_2, segment_label_2 = self.get_loss(data, '2', document, turn_nums)
        loss = (loss_1 + loss_2).mean()
        # For prediction_logits
        if evaluate:
            # eval_prediction_logits = torch.cat(eval_prediction_logits, 0)  
            eval_pred_logits_1 = self.prepare_eval(pred_logits_1, turn_nums)
            eval_pred_logits_2 = self.prepare_eval(pred_logits_2, turn_nums)

            return loss, eval_pred_logits_1, eval_pred_logits_2, segment_label_1, segment_label_2
        
        return loss
        

class SummarizerModel(nn.Module):
    def __init__(self, params, force_cpu=False):
        super().__init__()
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if force_cpu:
            DEVICE = torch.device('cpu')

        if params['load_path']:
            self.generator = BartForConditionalGeneration.from_pretrained(params['model_name'], state_dict = torch.load(params['load_path']))
        else:
            self.generator = BartForConditionalGeneration.from_pretrained(params['model_name'])
        
        last_layer = self.generator.model.encoder.layers[-1]
        self.generator.to(DEVICE)
        bart_config = BartConfig() 
        
        self.encoder_p1 = type(last_layer)(bart_config)
        self.encoder_p1.load_state_dict(last_layer.state_dict())
        self.encoder_p2 = type(last_layer)(bart_config)
        self.encoder_p2.load_state_dict(last_layer.state_dict())

        self.tokenizer = BartTokenizer.from_pretrained(params['model_name']) # Need to add base to "tokenization_bart.py" when using transformers==2.11.0
        self.config = self.generator.model.config
        if params['add_module_loss']:
            self.classifier = nn.Linear(self.generator.model.config.d_model, 7)
            self.classifier.to(DEVICE)
        elif params['add_functurn_loss']:
            self.classifier = nn.Linear(self.generator.model.config.d_model, 2)
            self.classifier.to(DEVICE)

        self.params = params
         
    def save(self, output_dir, name='pytorch.bin'):
        torch.save(self.state_dict(), os.path.join(output_dir, name))

    def forward(self, source_ids, source_mask, inference=False, kwargs={}):
        hidden_states, attention_mask= self.encode(input_ids=source_ids, attention_mask=source_mask)
        encoder_outputs_p1 = self.encode_head(hidden_states, attention_mask, self.encoder_p1)
        encoder_outputs_p2 = self.encode_head(hidden_states, attention_mask, self.encoder_p2)

        assert encoder_outputs_p1.last_hidden_state.shape == encoder_outputs_p2.last_hidden_state.shape

        if inference:
            return encoder_outputs_p1, encoder_outputs_p2
            #return self.generate(
            #    encoder_outputs_p1, encoder_outputs_p2,
            #    input_ids=source_ids, attention_mask=source_mask,
            #    num_beams=kwargs['num_beams'],
            #    max_length=kwargs['max_length'],
            #    no_repeat_ngram_size=kwargs['no_repeat_ngram_size'],
            #    early_stopping=True
            #)

        outputs_p1 = self.generator(input_ids=None,
                    attention_mask=source_mask,
                    encoder_outputs=encoder_outputs_p1,
                    decoder_input_ids=kwargs['target_ids_p1'],
                    labels=kwargs['target_labels_p1'])
        outputs_p2 = self.generator(input_ids=None,
                    attention_mask=source_mask,
                    encoder_outputs=encoder_outputs_p2,
                    decoder_input_ids=kwargs['target_ids_p2'],
                    labels=kwargs['target_labels_p2'])

        return outputs_p1, outputs_p2, encoder_outputs_p1.last_hidden_state, encoder_outputs_p2.last_hidden_state

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, num_beams=None, max_length=None, no_repeat_ngram_size=None, early_stopping=None):
        encoding_p1, encoding_p2 = self.forward(input_ids, attention_mask, inference=True)
        assert encoding_p1.last_hidden_state.shape == encoding_p2.last_hidden_state.shape
        batch_size = input_ids.shape[0]
        decoding_p1 = self.decode(encoding_p1, attention_mask, batch_size, input_ids,
            max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping
        )
        decoding_p2 = self.decode(encoding_p2, attention_mask, batch_size, input_ids,
            max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping
        )

        return decoding_p1, decoding_p2 

    @torch.no_grad()
    def encode_for_prediction(
        self,
        inputs: Optional[torch.Tensor] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.generator.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.generator.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.generator.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.generator.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.generator.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.generator.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.generator.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.generator.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generator.config.eos_token_id

        if eos_token_id is None and hasattr(self.generator.config, "decoder"):
            eos_token_id = self.generator.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.generator.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generator.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generator.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.generator.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self.generator._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.generator.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.generator._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.generator.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self.generator._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
        return model_kwargs

    @torch.no_grad()
    def decode(self, encoder_outputs, attention_mask, batch_size, input_ids, max_length=None, num_beams=None, no_repeat_ngram_size=None, early_stopping=None, **model_kwargs):
        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        num_beam_groups = self.config.num_beam_groups
        early_stopping = early_stopping if early_stopping is not None else self.generator.config.early_stopping
        model_kwargs['attention_mask'] = attention_mask
        model_kwargs['output_attentions'] = self.config.output_attentions
        model_kwargs['output_hidden_states'] = self.config.output_hidden_states,
        model_kwargs['uses_cache'] = self.config.use_cache,
        model_kwargs['encoder_outputs'] = encoder_outputs
        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = self.generator._prepare_decoder_input_ids_for_generation(
            #batch_size,
            input_ids=input_ids,
            decoder_start_token_id=None,
            bos_token_id=self.config.bos_token_id
            #model_kwargs=model_kwargs,
        )

        # 5. Prepare `max_length` depending on other stopping criteria
        # default to config if still None
        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and self.config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self.generator._get_logits_processor(
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=None,
            encoder_input_ids=input_ids,
            bad_words_ids=None,
            min_length=self.config.min_length,
            max_length=max_length,
            eos_token_id=self.config.eos_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=None,
            remove_invalid_values=None
         #   logits_processor= LogitsProcessorList(),
        )

        # 8. prepare stopping criteria
        stopping_criteria = self.generator._get_stopping_criteria(
            max_length=max_length, max_time=None#, stopping_criteria=StoppingCriteriaList()
        )

        if self.config.num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        #if stopping_criteria.max_length is None:
        #    raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 10. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=DEVICE,
            length_penalty=self.config.length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=self.config.num_return_sequences,
            max_length=max_length
        )

        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self.generator._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.generator.config.is_encoder_decoder, **model_kwargs
        )

        # 12. run beam search{
        return self.generator.beam_search(
            input_ids, beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            output_scores=self.config.output_scores,
            return_dict_in_generate=self.config.return_dict_in_generate,
            synced_gpus=False,
            **model_kwargs
        )

    def encode(self, input_ids=None, attention_mask=None):
        training = self.training

        # retrieve input_ids
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            raise ValueError("You have to specify input_ids")

        inputs_embeds = self.generator.model.encoder.embed_tokens(input_ids) * self.generator.model.encoder.embed_scale
        embed_pos = self.generator.model.encoder.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.generator.model.encoder.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, 
                            p=self.generator.model.encoder.dropout,
                            training=training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        output_attentions = None

        for idx, encoder_layer in enumerate(self.generator.model.encoder.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.generator.model.encoder.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            elif self.generator.model.encoder.config.gradient_checkpointing and training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    None,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=None,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]


        return hidden_states, attention_mask

    def encode_head(self, hidden_states, attention_mask, encoder_layer):
        output_attentions = self.config.output_attentions
        training = self.generator.model.encoder.training
        return_dict = self.config.use_return_dict

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if training and (dropout_probability < self.generator.model.encoder.layerdrop):  # skip the layer
            layer_outputs = (None, None)
#        elif self.generator.model.encoder.gradient_checkpointing and training:
        elif self.generator.model.encoder.config.gradient_checkpointing and training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(encoder_layer),
                hidden_states,
                attention_mask,
                None,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        if not return_dict:
            return tuple(v for v in [hidden_states] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states)


class ModelWrapper(nn.Module):
    def __init__(self, args, params):
        super().__init__()
        self.module = SummarizerModel(params)
        self.distributed = args.distributed
        if self.distributed:
            self.model = nn.DataParallel(self.model)
            self.module = self.model.module

    def forward(self, source_ids, source_mask, inference=False, **kwargs):
        if self.distributed:
            return self.model(source_ids, source_mask, inference, kwargs)
        return self.module(source_ids, source_mask, inference, kwargs)