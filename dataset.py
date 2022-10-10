import pytorch_lightning as pl
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, TensorDataset
from pytorch_lightning.trainer.supporters import CombinedLoader

""" Dataset Class """


class TextInputFeaturesDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.max_source_length, self.args.max_source_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].source_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True
        print("size of the attn_mask", attn_mask.size)
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask),)


class TextDatasetModule(pl.LightningDataModule):

    def __init__(self, train, val, dev, test, args):
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.dev_dataset = None
        self.train = train
        self.val = val
        self.dev = dev[1]
        self.test = test[1]

        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.args = args

    def setup(self, stage=None):
        self.train_dataset = TextInputFeaturesDataset(self.train, self.args)
        self.val_dataset = TextInputFeaturesDataset(self.val, self.args)
        self.dev_dataset = TextInputFeaturesDataset(self.dev, self.args)
        self.test_dataset = TextInputFeaturesDataset(self.test, self.args)

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_batch_size // self.args.gradient_accumulation_steps,
            num_workers=4
        )

    def val_dataloader(self):
        eval_sampler_loss = SequentialSampler(self.val_dataset)
        loader_a = DataLoader(
            self.val_dataset,
            sampler=eval_sampler_loss,
            batch_size=self.eval_batch_size,
            num_workers=4
        )

        eval_sampler_bleu = SequentialSampler(self.dev_dataset)
        loader_b = DataLoader(
            self.dev_dataset,
            sampler=eval_sampler_bleu,
            batch_size=self.eval_batch_size,
            num_workers=4
        )
        loaders = {"loss": loader_a, "bleu": loader_b}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def test_dataloader(self):
        test_sampler = SequentialSampler(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            sampler=test_sampler,
            num_workers=4
        )
