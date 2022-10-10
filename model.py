# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import copy
import numpy as np
import pytorch_lightning as pl
from beam import *
from bleu import _bleu, compute_bleu
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids,source_mask,position_idx,attn_mask,target_ids=None,target_mask=None,args=None):   
        # embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(source_ids)
        attn_mask = attn_mask[:,:len(nodes_mask[1]),:len(nodes_mask[1])]
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]  
        
        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        # source_mask=token_mask.float()
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds

        # Defining the ligthining model


class Seq2SeqPredictor(pl.LightningModule):

    def __init__(self, model, tokenizer, targets, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = args.learning_rate
        self.eps = args.adam_epsilon
        self.args = args
        self.dev_examples_target, self.test_examples_target = targets

    def forward(self, x):
        # need to edit it and return accordingly

        output = self.model(x)

        return output

    def training_step(self, batch, batch_idx):  # called everytime a training needs to occur
        # print("I am called training step")
        source_ids, source_mask, position_idx, att_mask, target_ids, target_mask = batch
        loss, _, _ = self.model(source_ids, source_mask, position_idx, att_mask, target_ids, target_mask)

        self.log("train_loss", loss, prog_bar=False, logger=True)

        return {"loss": loss}  # returns a dictionary

    def validation_step(self, batch, batch_idx):  # called everytime a validation needs to occur
        # print("I am called validation step")
        # for loss function
        source_ids, source_mask, position_idx, att_mask, target_ids, target_mask = batch["loss"]
        _, loss, num = self.model(source_ids, source_mask, position_idx, att_mask, target_ids, target_mask)

        # for bleu prediction
        source_ids, source_mask, position_idx, att_mask, target_ids, target_mask = batch["bleu"]
        preds = self.model(source_ids, source_mask, position_idx, att_mask)

        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)

        return {"loss": loss, "num": num, "preds": preds}

    def test_step(self, batch, batch_idx):  # called everytime a test needs to occur
        # for bleu prediction
        source_ids, source_mask, position_idx, att_mask, target_ids, target_mask = batch
        preds = self.model(source_ids, source_mask, position_idx, att_mask)

        return {"preds": preds}

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)
        # automatically find the total number of steps we need!
        # num_training_steps, num_warmup_steps = self.compute_warmup(self.num_training_steps, num_warmup_steps=0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches * 0.1,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("avg_train_loss", avg_loss, logger=True)

    def validation_epoch_end(self, validation_step_outputs):
        total_loss = torch.stack([x['loss'] for x in validation_step_outputs]).sum()
        tokens_num = torch.stack([x['num'] for x in validation_step_outputs]).sum()
        avg_loss = total_loss / tokens_num

        preds = torch.stack([x['preds'] for x in validation_step_outputs])
        examples = self.dev_examples_target
        p = []
        for pred in preds:
            print("the prediction are:____________", pred)
            t = pred[0][0].cpu().numpy()
            t = list(t)
            print("the t list is ***********************", t)
            if 0 in t:
                t = t[:t.index(0)]
            text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
            p.append(text)

        print("the the text list is ***********************", text)
        # calculate the bleu score
        predictions_list = []
        accs = []
        target_list = []
        # with open(os.path.join(self.args.output_dir, "dev.output"), 'w') as f, open(os.path.join(self.args.output_dir, "dev.gold"), 'w') as f1:
        for ref, gold in zip(p, examples):
            print("ref is",ref.strip().split())
            print("gold is",gold.target.strip().split())
            predictions_list.append(ref.strip().split())
            target_list.append(gold.target.strip().split())
            # f.write(ref + '\n')
            # f1.write(gold.target + '\n')
            accs.append(ref == gold.target)

        smooth = True
        max_order = 4
        bleu_score, _, _, _, _, _ = compute_bleu(predictions_list, target_list, max_order, smooth)
        dev_bleu = round(100 * bleu_score, 2)
        dev_bleu = round(dev_bleu,2)
        # dev_bleu = round(_bleu(os.path.join(self.args.output_dir, "dev.gold"), os.path.join(self.args.output_dir, "dev.output")), 2)
        xmatch = round(np.mean(accs) * 100, 4)
        self.log("avg_val_bleu-4", dev_bleu, logger=True)
        self.log("xMatch", round(np.mean(accs) * 100, 4), logger=True)
        self.log("avg_val_loss", avg_loss, logger=True)

        # calculate the accuracy score

    def test_epoch_end(self, test_step_outputs):  # the average pre@ k
        preds = torch.stack([x['preds'] for x in test_step_outputs])
        examples = self.test_examples_target
        p = []
        for pred in preds:
            t = pred[0][0].cpu().numpy()
            t = list(t)
            if 0 in t:  # it was only t before
                t = t[:t.index(0)]
                text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)

        # calculate the bleu score
        predictions = []
        accs = []
        with open(os.path.join(self.args.output_dir, "dev.output"), 'w') as f, open(
                os.path.join(self.args.output_dir, "dev.gold"), 'w') as f1:
            for ref, gold in zip(p, examples):
                predictions.append(ref)
                f.write(ref + '\n')
                f1.write(gold.target + '\n')
                accs.append(ref == gold.target)

        dev_bleu = round(
            _bleu(os.path.join(self.args.output_dir, "dev.gold"), os.path.join(self.args.output_dir, "dev.output")), 2)
        xmatch = round(np.mean(accs) * 100, 4)
        self.log("avg_test_bleu-4", dev_bleu, logger=True)
        self.log("xMatch", round(np.mean(accs) * 100, 4), logger=True)

    # what about on_test_end

