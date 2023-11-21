import os
import sys

from time import strftime, localtime
import logging

model_name = 'CM_BEIT'
img_dir = '/hy-tmp/data/dataset_image'
check_point_path = '/hy-tmp/models'


log_file = f'/root/logs/{model_name}-{strftime("%y%m%d-%H%M", localtime())}.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(log_file))


import pickle, json
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from timm.data.transforms import RandomResizedCropAndInterpolation

import utils
from randaug import RandomAugment

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = 'center'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask
        # data["num_tokens"] = num_tokens

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


class SarcasmDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("sarcasm.train.jsonl", )
        elif split == "val":
            return ("sarcasm.test.jsonl", )
        elif split == "test":
            return ("sarcasm.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["label"] = self.items[index]["label"]
        data["image_id"] = int(self.items[index]["image_id"])
        return data

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, image_path, ocr_path=None):
        image_names = os.listdir(image_path)
        if ocr_path:
            with open(ocr_path,'rb') as fin:
                text_in_imgs = pickle.load(fin)
        
        items = []
        index_file = os.path.join(data_path, f"sarcasm.train.jsonl")
        with open(os.path.join(data_path, "train.txt"),'r',encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [x.strip() for x in lines]
            for i in range(len(lines)):
                line = lines[i]
                data = eval(line)
                img_id,text,label = data
                if img_id+'.jpg' in image_names:
                    # text = text.replace("#", '')
                    text_in_img = text_in_imgs[img_id] if ocr_path else ''
                    if text_in_img:
                        tokens = tokenizer.tokenize(text) + [tokenizer.sep_token]
                        tokens += tokenizer.tokenize(text_in_img)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    else:
                        tokens = tokenizer.tokenize(text)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                            "image_path": os.path.join(image_path, f"{img_id}.jpg"),
                            "text_segment": token_ids,
                            "image_id": img_id,
                            "label": label,
                        })
        _write_data_into_jsonl(items, index_file)
        items = []
        index_file = os.path.join(data_path, f"sarcasm.valid.jsonl")
        with open(os.path.join(data_path, "valid2.txt"),'r',encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [x.strip() for x in lines]
            for i in range(len(lines)):
                line = lines[i]
                data = eval(line)
                img_id,text,label1,label = data
                if img_id+'.jpg' in image_names:
                    # text = text.replace("#", '')
                    text_in_img = text_in_imgs[img_id] if ocr_path else ''
                    if text_in_img:
                        # encoded_dict = tokenizer(
                        #                 text,                      # Sentence to encode.
                        #                 text_in_img,
                        #                 add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        #                 padding = 'max_length',
                        #                 truncation = 'longest_first',
                        #                 max_length = 64,    # Pad & truncate all sentences.
                        #                 return_attention_mask = True,   # Construct attn. masks.
                        #                 return_length = True,
                        #         )
                        # token_ids = encoded_dict.input_ids
                        tokens = tokenizer.tokenize(text) + [tokenizer.sep_token]
                        tokens += tokenizer.tokenize(text_in_img)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    else:
                        tokens = tokenizer.tokenize(text)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                            "image_path": os.path.join(image_path, f"{img_id}.jpg"),
                            "text_segment": token_ids,
                            "image_id": img_id,
                            "label": label,
                        })
        _write_data_into_jsonl(items, index_file)
        items = []
        index_file = os.path.join(data_path, f"sarcasm.test.jsonl")
        with open(os.path.join(data_path, "test2.txt"),'r',encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [x.strip() for x in lines]
            for i in range(len(lines)):
                line = lines[i]
                data = eval(line)
                img_id,text,label1,label = data
                if img_id+'.jpg' in image_names:
                    # text = text.replace("#", '')
                    text_in_img = text_in_imgs[img_id] if ocr_path else ''
                    if text_in_img:
                        # encoded_dict = tokenizer(
                        #                 text,                      # Sentence to encode.
                        #                 text_in_img,
                        #                 add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        #                 padding = 'max_length',
                        #                 truncation = 'longest_first',
                        #                 max_length = 64,    # Pad & truncate all sentences.
                        #                 return_attention_mask = True,   # Construct attn. masks.
                        #                 return_length = True,
                        #         )
                        # token_ids = encoded_dict.input_ids
                        tokens = tokenizer.tokenize(text) + [tokenizer.sep_token]
                        tokens += tokenizer.tokenize(text_in_img)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    else:
                        tokens = tokenizer.tokenize(text)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                            "image_path": os.path.join(image_path, f"{img_id}.jpg"),
                            "text_segment": token_ids,
                            "image_id": img_id,
                            "label": label,
                        })
        _write_data_into_jsonl(items, index_file)


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0), interpolation=args.train_interpolation), 
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


task2dataset = {
    # "nlvr2": NLVR2Dataset, 
    # "vqav2": VQAv2Dataset, 
    # "flickr30k": RetrievalDataset, 
    # "coco_retrieval": RetrievalDataset,  
    # "coco_captioning": CaptioningDataset,
    # "nocaps": CaptioningDataset,
    # "imagenet": ImageNetDataset,
    "sarcasm": SarcasmDataset,
}


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    dataset_class = task2dataset[args.task]
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, tokenizer=tokenizer, 
        num_max_bpe_tokens=args.num_max_bpe_tokens, 
        task=args.task, **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )


def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=True)


from datasets import SarcasmDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/root/code/beit3/beit3.spm")

SarcasmDataset.make_dataset_index(
    data_path="/hy-tmp/data/data-of-multimodal-sarcasm-detection/text",
    tokenizer=tokenizer,
    image_path="/hy-tmp/data/dataset_image",
    ocr_path="/hy-tmp/data/str_in_images_cleaned2.pkl",
)


# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    VisionEmbedding,
)
from torchscale.component.multiway_network import MutliwayEmbedding


class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out


import math
import torch
import torch.nn as nn

from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TwoLayerMLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class attentional_pooling(nn.Module):
    def __init__(self, embed_dim, norm_layer):
        super().__init__()
        self.query = nn.Parameter(torch.rand(10, embed_dim))
        self.norm = norm_layer(embed_dim)
        # self.dense = nn.Linear(embed_dim, embed_dim)
        # self.activation = nn.Tanh()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 2, batch_first=True)


    def forward(self, x, key_padding_mask):
        bs = x.shape[0]
        attn_output, attn_output_weights = self.multihead_attn(self.query.repeat(bs, 1, 1), x, x, key_padding_mask=key_padding_mask)
        cls_rep = self.norm(cls_rep)
        cls_rep = torch.mean(attn_output, 1)
        # pooled_output = self.dense(cls_rep)
        # pooled_output = self.activation(pooled_output)

        return cls_rep


class Sarcasm_Head(nn.Module):
    def __init__(self, embed_dim, num_classes, norm_layer):
        super().__init__()
        self.pooler = attentional_pooling(embed_dim, norm_layer)
        self.mlp = TwoLayerMLP(
            in_features = embed_dim,
            hidden_features = embed_dim * 2,
            out_features = num_classes,
            norm_layer = norm_layer,
            norm_input=False,
        )

    def forward(self, x, key_padding_mask):
        pooled_output = self.pooler(x, key_padding_mask)
        return self.mlp(pooled_output)

# class BEiT3ForSarcasmDetection(BEiT3Wrapper):
#     def __init__(
#             self, 
#             args, 
#             num_classes, 
#             norm_layer=nn.LayerNorm, 
#             **kwargs
#     ):
#         super(BEiT3ForSarcasmDetection, self).__init__(args=args)
#         embed_dim = args.encoder_embed_dim
#         self.head = Sarcasm_Head(embed_dim=embed_dim, num_classes=2, norm_layer=norm_layer)
#         self.head.apply(self._init_weights)

#     def forward(self, image, text, padding_mask, **kwargs):
#         outputs = self.beit3(
#             textual_tokens=text, 
#             visual_tokens=image, 
#             text_padding_position=padding_mask, 
#         )
#         # encoder_out encoder_embedding encoder_padding_mask encoder_states l_aux multiway_split_position
#         x = outputs["encoder_out"]
#         return self.head(x, outputs["encoder_padding_mask"].bool()), outputs

class BEiT3ForSarcasmDetection(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForSarcasmDetection, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

        # for name, param in self.named_parameters():
        #     if "pooler" in name or "head" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False


    def forward(self, image, text, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=text, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
        # encoder_out encoder_embedding encoder_padding_mask encoder_states l_aux multiway_split_position
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        return self.head(cls_rep), outputs


from timm.models.registry import register_model
@register_model
def beit3_base_patch16_224_sarcasm(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForSarcasmDetection(args, num_classes=2, **kwargs)
    return model


from typing import Iterable, Optional
from sklearn import metrics
from timm.utils import ModelEma


class TaskHandler(object):
    def __init__(self) -> None:
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, data_loader, **kwargs):
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()

class SarcasmHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.predictions = []
        self.tokenizer = get_sentencepiece_model_for_beit3(args)
        # self.num_beams = args.num_beams
        self.max_len = args.num_max_bpe_tokens
        # self.length_penalty = args.length_penalty
        self.vocab_size = args.vocab_size
        self.supconloss = utils.SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)

    def train_batch(self, model, image, language_tokens, padding_mask, label, image_id):
        bsz = label.shape[0]
        logits, outputs = model(
            image=image, text=language_tokens, 
            padding_mask=padding_mask)
        celoss = self.criterion(logits, label)
        # print(outputs['encoder_embedding'].shape)
        # print(outputs['encoder_padding_mask'].shape)
        # print(outputs, label)
        feature = outputs['encoder_embedding']
        multiway_split_position = outputs["multiway_split_position"]

        vision_cls = feature[:, 0, :]
        language_cls = feature[:, multiway_split_position, :]
        feature = torch.cat((vision_cls.unsqueeze(1), language_cls.unsqueeze(1)), dim=1)

        # feature = outputs['encoder_embedding']
        feature = outputs['encoder_embedding'] * (1 - outputs['encoder_padding_mask'].unsqueeze(-1).type_as(outputs['encoder_embedding']))
        # feature = outputs['encoder_out']
        embedding_dim = feature.shape[-1]
        # feature = F.layer_norm(feature, (embedding_dim,))
        feature = F.normalize(feature, dim=-1)
        scloss = self.supconloss(feature, label)
        # print(celoss, scloss)
        cost = celoss + scloss
        return {
            "loss": celoss,
        }
    
    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()

    def eval_batch(self, model, image, language_tokens, padding_mask, label, image_id):
        logits, outputs = model(
            image=image, text=language_tokens, 
            padding_mask=padding_mask)
        bs = language_tokens.shape[0]

        self.predictions.append({
                "logits": logits.cpu(),
                "label": label.cpu(),
                "image_id": image_id.cpu(),
            })
    
    def after_eval(self, **kwargs):
        logits_all, labels_all, img_id_all = None, None, None
        for b in self.predictions:
            if logits_all is None:
                logits_all = b['logits']
                labels_all = b['label']
                img_id_all = b['image_id']
            else:
                logits_all = torch.cat((logits_all, b['logits']), dim=0)
                labels_all = torch.cat((labels_all, b['label']), dim=0)
                img_id_all = torch.cat((img_id_all, b['image_id']), dim=0)
        
        f1 = metrics.f1_score(labels_all, torch.argmax(logits_all, -1))
        precision = metrics.precision_score(labels_all, torch.argmax(logits_all, -1))
        recall = metrics.recall_score(labels_all, torch.argmax(logits_all, -1))
        acc = metrics.accuracy_score(labels_all, torch.argmax(logits_all, -1))
        logger.info(f"acc {acc:4f}", f"precision {precision:4f}", f"recall {recall:4f}", f"f1 {f1:4f}")

        preds_all = torch.argmax(logits_all, -1).tolist()
        logits_all = logits_all.tolist()
        labels_all = labels_all.tolist()
        img_id_all = img_id_all.tolist()
        result = []
        for logit, pred, label, image_id in zip(logits_all, preds_all, labels_all, img_id_all):
            result.append({'logit':logit, "prediction":pred, 'label':label, "image_id":image_id, })
        self.predictions = result
        return result
    

def get_handler(args):
    # if args.task == "nlvr2":
    #     return NLVR2Handler()
    # elif args.task == "vqav2":
    #     return VQAHandler()
    # elif args.task in ("flickr30k", "coco_retrieval"):
    #     return RetrievalHandler()
    # elif args.task in ("coco_captioning", "nocaps"):
    #     return CaptioningHandler(args)
    # elif args.task in ("imagenet"):
    #     return ImageNetHandler(args)
    # elif args.task in ("sarcasm"):
    return SarcasmHandler(args)
    # else:
    #     raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: TaskHandler, epoch: int, start_steps: int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer: Optional[logging.Logger] = None, 
        task = None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        # mixup for imagenet finetuning
        if mixup_fn is not None:
            data["image"], data["label"] = mixup_fn(data["image"], data["label"])
        
        if task in ["coco_captioning", "nocaps"]:
            data["global_step"] = global_step

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "epoch": epoch,
                "step": f"{step}/{len(data_loader)}",
                "loss": loss_value, 
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.debug(kwargs)
            if global_step % print_freq == 0 or global_step == len(data_loader) - 1:
                log_writer.info(kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.debug(kwargs)
            if global_step % print_freq == 0 or global_step == len(data_loader) - 1:
                log_writer.info(kwargs)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler, log_writer: Optional[logging.Logger] = None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    results = handler.after_eval()

    if log_writer is not None:
        kwargs = {
            "header": header,
        }
        for key in results:
            kwargs[key] = results[key]
        log_writer.info(kwargs)

    return results

# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit

# from engine_for_finetuning import train_one_epoch, get_handler, evaluate
# from datasets import create_downstream_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps', 'imagenet', 'sarcasm'], 
                        help='Name of task to fine-tuning')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--checkpoint_activations', action='store_true', default=True, 
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str, required=True, 
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # parameter for imagenet finetuning
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # augmentation parameters for imagenet finetuning
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # evaluation parameters for imagenet
    parser.add_argument('--crop_pct', type=float, default=None)

    # random Erase params for imagenet finetuning
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # parameter for captioning finetuning
    parser.add_argument('--captioning_mask_prob', type=float, default=0.6)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    args = [
    '--model', 'beit3_base_patch16_224',
    '--input_size', '224',
    '--task', 'sarcasm',
    '--batch_size', '16',
    '--layer_decay', '1.0',
    '--lr', '2e-5',
    '--update_freq', '1',
    '--randaug',
    '--epochs', '10',
    '--warmup_epochs', '1',
    '--drop_path', '0.1',
    '--sentencepiece_model', '/root/code/beit3/beit3.spm',
    '--finetune', '/hy-tmp/models/beit3_base/beit3_base_patch16_224.pth',
    '--data_path', '/hy-tmp/data/data-of-multimodal-sarcasm-detection/text',
    '--output_dir', '/hy-tmp/models/best_state',
    '--log_dir', '/root/logs',
    '--weight_decay', '0.01',
    '--seed', '42',
    '--save_ckpt_freq', '5',
    '--task_head_lr_weight', '20.0',
    '--opt_betas', '0.9', '0.98',
    ]

    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        log_writer = logger
    else:
        log_writer = None

    data_loader_train, data_loader_val = create_downstream_dataset(args)

    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    logger.info("model_config = %s" % model_config)
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )

    if args.finetune:
        utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        logger.info("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model = %s" % str(model_without_ddp))
    logger.info('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    logger.info("LR = %.8f" % args.lr)
    logger.info("Batch size = %d" % total_batch_size)
    logger.info("Update frequent = %d" % args.update_freq)
    logger.info("Number of training examples = %d" % len(data_loader_train.dataset))
    logger.info("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(lrs)
    elif args.task_head_lr_weight > 1:
        assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
    else:
        assigner = None

    if assigner is not None:
        logger.info("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    if args.distributed:
        torch.distributed.barrier()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        logger.info("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    task_handler = get_handler(args)

    # mixup for imagenet
    mixup_fn = None
    if args.task in ["imagenet", "in1k"]:
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            logger.info("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=args.nb_classes)

    if args.eval:
        data_loader_test = create_downstream_dataset(args, is_eval=True)
        if args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet"]:
            ext_test_stats, task_key = evaluate(data_loader_test, model, device, task_handler)
            logger.info(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
            exit(0)
        elif args.task == "vqav2":
            result, _ = evaluate(data_loader_test, model, device, task_handler)
            utils.dump_predictions(args, result, "vqav2_test")
            exit(0)
        elif args.task in ["coco_captioning", "nocaps"]:
            predictions, _ = evaluate(data_loader_test, model, device, task_handler)
            prediction_file = utils.dump_predictions(args, predictions, "{}_test".format(args.task))
            if utils.is_main_process() and args.task == "coco_captioning":
                captioning_result = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_test".format(args.task))
                result_file = os.path.join(args.output_dir, f"{args.task}_result.json")
                logger.info(json.dumps(captioning_result))
                utils.write_result_to_jsonl(captioning_result, result_file)
            exit(0)
        elif args.task == "sarcasm":
            predictions = evaluate(data_loader_test, model, device, task_handler)
            result_file = os.path.join(args.output_dir, f"{args.task}_test.json")
            with open(result_file, "w") as fp:
                json.dump(predictions, fp, indent=2)
            exit(0)

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # if log_writer is not None:
        #     log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, task_handler, epoch, 
            epoch * num_training_steps_per_epoch, lr_schedule_values, loss_scaler, 
            args.clip_grad, args.update_freq, model_ema, log_writer, args.task, mixup_fn,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            if args.task not in ["coco_captioning", "nocaps", "sarcasm"]:
                test_stats, task_key = evaluate(data_loader_val, model, device, task_handler)
            elif args.task == "sarcasm":
                predictions = evaluate(data_loader_val, model, device, task_handler)
                logits_all, labels_all, img_id_all = None, None, None
                for b in predictions:
                    if logits_all is None:
                        logits_all = [b['logit']]
                        labels_all = [b['label']]
                        img_id_all = [b['image_id']]
                    else:
                        logits_all += [b['logit']]
                        labels_all += [b['label']]
                        img_id_all += [b['image_id']]
                
                logits_all = np.array(logits_all)
                predicion_all = np.argmax(logits_all, axis=1)

                f1 = metrics.f1_score(labels_all, predicion_all)
                precision = metrics.precision_score(labels_all, predicion_all)
                recall = metrics.recall_score(labels_all, predicion_all)
                acc = metrics.accuracy_score(labels_all, predicion_all)
                test_stats = {
                    "acc": acc,
                    "f1": f1,
                    "recall": recall,
                    "precision": precision,
                }
                logger.info(f"acc {acc:4f}", f"precision {precision:4f}", f"recall {recall:4f}", f"f1 {f1:4f}")
                task_key = "acc"
            else:
                predictions, _ = evaluate(data_loader_val, model, device, task_handler)
                prediction_file = utils.dump_predictions(args, predictions, f"{args.task}_val_e{epoch}")
                result_file = os.path.join(args.output_dir, f"{args.task}_result_val_e{epoch}.json")
                task_key = "CIDEr"
                if utils.is_main_process():
                    test_stats = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_val".format(args.task))
                    utils.write_result_to_jsonl(test_stats, result_file)
                torch.distributed.barrier()
                if not utils.is_main_process():
                    test_stats = utils.read_result_from_jsonl(result_file)

            logger.info(f"Performance of the network on the {len(data_loader_val.dataset)} val images: {test_stats[task_key]:.1f}%")
            if max_accuracy < test_stats[task_key]:
                max_accuracy = test_stats[task_key]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            logger.info(f'Max performance: {max_accuracy:.2f}%')
            if log_writer is not None:
                # log_writer.update(acc=test_stats[task_key], head="perf", step=epoch)
                log_writer.info(f'epoch:{epoch} test_stats:{test_stats}')
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         # **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            logger.info(log_stats)
            # if log_writer is not None:
            #     log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
