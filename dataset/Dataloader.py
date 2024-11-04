import os

from .VideoLoader import  load_batch_video
from .FeatureLoader import load_batch_feature
from .Dataset import build_dataset
import torch


def collate_fn_(inputs, data_cfg, task, is_train, 
    text_tokenizer=None, gloss_tokenizer=None,name2keypoint=None):
    outputs = {
        'name':[i['name'] for i in inputs],
        'gloss':[i.get('gloss','') for i in inputs],
        'text':[i.get('text','') for i in inputs],
        'num_frames':[i['num_frames'] for i in inputs]}
    if task == 'S2G':
        outputs['recognition_inputs'] = gloss_tokenizer(outputs['gloss'])

        sgn_videos, sgn_keypoints, sgn_lengths = load_batch_video(
            zip_file=data_cfg['zip_file'],
            names=outputs['name'],
            num_frames=outputs['num_frames'],
            transform_cfg=data_cfg['transform_cfg'],
            dataset_name=data_cfg['dataset_name'], 
            pad_length=data_cfg.get('pad_length','pad_to_max'),
            pad = data_cfg.get('pad','replicate'),
            is_train=is_train,  
            name2keypoint=name2keypoint,
            )
        outputs['recognition_inputs']['sgn_videos'] = sgn_videos
        outputs['recognition_inputs']['sgn_keypoints'] = sgn_keypoints
        outputs['recognition_inputs']['sgn_lengths'] = sgn_lengths


    if task in ['S2T','G2T','S2T_Ensemble']:
        tokenized_text = text_tokenizer(input_str=outputs['text'])
        outputs['translation_inputs'] = {**tokenized_text}
        if task == 'S2T':
            outputs['recognition_inputs'] = gloss_tokenizer(outputs['gloss'])
            outputs['translation_inputs']['gloss_ids'] = outputs['recognition_inputs']['gloss_labels']  
            outputs['translation_inputs']['gloss_lengths'] = outputs['recognition_inputs']['gls_lengths'] 
            for feature_name in ['sgn_features', 'head_rgb_input','head_keypoint_input']:
                if feature_name in inputs[0]:
                    outputs['recognition_inputs'][feature_name], sgn_mask, sgn_lengths = \
                        load_batch_feature(features=[i[feature_name]+1.0e-8 for i in inputs])
            outputs['recognition_inputs']['sgn_mask'] = sgn_mask
            outputs['recognition_inputs']['sgn_lengths'] = sgn_lengths
        elif task == 'G2T':
            tokenized_gloss = gloss_tokenizer(batch_gls_seq=outputs['gloss'])
            outputs['translation_inputs']['input_ids'] = tokenized_gloss['input_ids']
            outputs['translation_inputs']['attention_mask'] = tokenized_gloss['attention_mask']
        elif task == 'S2T_Ensemble':
            outputs['translation_inputs']['inputs_embeds_list'] = []
            outputs['translation_inputs']['attention_mask_list'] = []
            for ii in range(len(inputs[0]['inputs_embeds_list'])):
                inputs_embeds, mask_ ,_= load_batch_feature(features=[i['inputs_embeds_list'][ii] for i in inputs])
                outputs['translation_inputs']['inputs_embeds_list'].append(inputs_embeds)
                outputs['translation_inputs']['attention_mask_list'].append(mask_)
    return outputs

def build_dataloader(cfg, split, 
    text_tokenizer=None, gloss_tokenizer=None, 
    mode='auto', val_distributed=False):
    dataset = build_dataset(cfg['data'], split)
    mode = split if mode=='auto' else mode
    import  torch.distributed as dist
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='29500'
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl',rank=0,world_size=1)
    if mode=='train':

        # 分布式采样器，在数据并行训练中使用，确保每个步骤获得不同的采样结果
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            shuffle=cfg['training']['shuffle'] and split=='train'
        )
    else:
        if val_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    # 对数据进行 batch 的划分。
    # 数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    # 在训练模型时使用到此函数，用来 把训练数据分成多个小组 ，此函数每次抛出一组数据 。直至把所有的数据都抛出。就是做一个数据的初始化。
    dataloader = torch.utils.data.DataLoader(dataset,collate_fn=lambda x:collate_fn_(
                                                 inputs=x,
                                                 task=cfg['task'],
                                                 data_cfg=cfg['data'],
                                                 is_train=(mode=='train'),
                                                 text_tokenizer=text_tokenizer,
                                                 gloss_tokenizer=gloss_tokenizer,
                                                 name2keypoint=dataset.name2keypoints),
                                             batch_size=cfg['training']['batch_size'],
                                             num_workers=cfg['training'].get('num_workers',2),
                                             sampler=sampler,
                                             )
    return dataloader, sampler