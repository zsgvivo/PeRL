import math
import torch
import numpy as np
import random
from PIL import Image
import torchvision.transforms as T
# from verl.workers.actor.config import ActorConfig
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

def randomly_swap_images(images, seed=42):
    random.seed(seed)
    img_num = len(images)
    from copy import deepcopy
    new_images = deepcopy(images)
    shuffled_list = list(range(img_num))
    random.shuffle(shuffled_list)
    for j in range(img_num):
        new_images[j] = images[shuffled_list[j]]
    return new_images, shuffled_list

def swap_image_in_batch(batch, processor, config, step=0, total_steps=1, seed=42, decay_mode='linear', decay_coef=0.5, warmup_ratio=0.1):
    if "multi_modal_data" not in batch.non_tensor_batch:
        return batch, 0
    norm_step = step / total_steps
    
    # 计算 warm-up 步数
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Warm-up 阶段：线性增加 decay 从 0 到 1
    if warmup_steps > 0 and step < warmup_steps:
        warmup_factor = step / warmup_steps
    else:
        warmup_factor = 1.0
    
    # 计算基础 decay
    if decay_mode == 'exp':
        decay = 1.0 - decay_coef ** (total_steps - step)
    elif decay_mode == 'pow':
        decay = 1.0 - norm_step ** decay_coef
    elif decay_mode == 'linear':
        decay = 1.0 - norm_step
    else:
        decay = 1.0  # 默认不衰减
    
    # 应用 warm-up
    decay *= warmup_factor

    from copy import deepcopy
    new_batch = deepcopy(batch)
    # import pdb; pdb.set_trace()
    # print(new_batch.non_tensor_batch["reward_model"])
    # random seed
    random.seed(seed)
    for i, item in enumerate(new_batch.non_tensor_batch["multi_modal_data"]):
        if len(item['image']) < 2:
            continue
        if random.random() > decay: # Only perform swapping with probability equal to decay
            continue
        data_source = new_batch.non_tensor_batch["data_source"][i]
        if data_source == 'IIGroup/Mantis-Instruct-nlvr2':
            assert len(item['image']) == 2, f"Unsupported image number: {len(item['image'])}, IIGroup/Mantis-Instruct-nlvr2 data should have 2 images"
            new_batch.non_tensor_batch['multi_modal_data'][i]['image'], shuffled_idx = randomly_swap_images(
                item['image'],
                seed=seed
            )
            if new_batch.non_tensor_batch['change_needed'][i]:
                if shuffled_idx != list(range(len(item['image']))):
                    assert new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B'], f"Unsupported ground_truth: {new_batch.non_tensor_batch['reward_model'][i]['ground_truth']}"
                    new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = 'B' if new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] == 'A' else 'A'
        elif data_source == 'Mantis-Instruct-dreamsim':
            assert len(item['image']) == 3, f"Unsupported image number: {len(item['image'])}, Mantis-Instruct-dreamsim data should have 3 images"
            new_batch.non_tensor_batch['multi_modal_data'][i]['image'][1:], shuffled_idx = randomly_swap_images(
                item['image'][1:],
                seed=seed
            )   # only swap images in choices
            if new_batch.non_tensor_batch['change_needed'][i]:
                if shuffled_idx != list(range(len(item['image'][1:]))):
                    assert new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B'], f"Unsupported ground_truth: {new_batch.non_tensor_batch['reward_model'][i]['ground_truth']}"
                    new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = 'B' if new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] == 'A' else 'A'
        elif data_source=='Mantis-Instruct-iconqa':
            assert len(item['image']) <= 6, f"Unsupported image number: {len(item['image'])}, Mantis-Instruct-iconqa data should have no more than 6 images"
            new_batch.non_tensor_batch['multi_modal_data'][i]['image'][1:], shuffled_idx = randomly_swap_images(
                item['image'][1:],
                seed=seed
            )   # only swap images in choices
            if new_batch.non_tensor_batch['change_needed'][i]:
                if shuffled_idx != list(range(len(item['image'][1:]))):
                    assert new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B', 'C', 'D', 'E'], f"Unsupported ground_truth: {new_batch.non_tensor_batch['reward_model'][i]['ground_truth']}"
                    origin_ans_idx = ['A', 'B', 'C', 'D', 'E'].index(new_batch.non_tensor_batch["reward_model"][i]["ground_truth"])
                    new_ans_idx = shuffled_idx.index(origin_ans_idx)
                    new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = ['A', 'B', 'C', 'D', 'E'][new_ans_idx]
        elif data_source=='ShowHowTo':
            # import pdb; pdb.set_trace()
            def get_option_img_num(text):
                import re
                cnt = 0
                for match in re.finditer(r'<image>', text.split('Options:')[-1]):
                    cnt += 1
                return cnt
            if new_batch.non_tensor_batch['question_type'][i] in ['description_to_frame', 'missing_frame', 'next_frame']:
                option_img_num = get_option_img_num(new_batch.non_tensor_batch['extra_info'][i]['question'])
                assert option_img_num == 4, f"Unsupported image number: {option_img_num}, ShowHowTo data should have 4 images in options"
                new_batch.non_tensor_batch['multi_modal_data'][i]['image'][-4:], shuffled_idx = randomly_swap_images(
                    item['image'][-4:],
                    seed=seed
                )   # only swap images in choices
                if new_batch.non_tensor_batch['change_needed'][i]:
                    if shuffled_idx!= list(range(len(item['image'][-4:]))):
                        assert new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B', 'C', 'D'], f"Unsupported ground_truth: {new_batch.non_tensor_batch['reward_model'][i]['ground_truth']}"
                        origin_ans_idx = ['A', 'B', 'C', 'D'].index(new_batch.non_tensor_batch["reward_model"][i]["ground_truth"])
                        new_ans_idx = shuffled_idx.index(origin_ans_idx)
                        new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = ['A', 'B', 'C', 'D'][new_ans_idx]
            elif new_batch.non_tensor_batch['question_type'][i] == 'ordering':
                option_img_num = get_option_img_num(new_batch.non_tensor_batch['extra_info'][i]['question'])
                assert option_img_num == 0, f"Unsupported image number: {option_img_num}, ShowHowTo ordering data should have 0 images in options"
                # extract correct order
                import re
                def extract_options(text):
                    options = {}
                    lines = text.split('\n')
                    capture = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('Options:'):
                            capture = True
                            continue
                        if capture:
                            if not stripped:
                                break
                            match = re.match(r'^([A-D])\.\s*(.*)$', stripped)
                            if match:
                                key = match.group(1)
                                sequence = list(map(int, match.group(2).split('→')))
                                options[key] = sequence
                    return options
                options = extract_options(new_batch.non_tensor_batch['extra_info'][i]['question'])
                correct_order = options[new_batch.non_tensor_batch["reward_model"][i]["ground_truth"]]
                new_answer = random.choice(['A', 'B', 'C', 'D'])
                new_order = options[new_answer]
                new_images = deepcopy(item['image'])
                for j in range(len(correct_order)):
                    new_images[new_order[j]-1] = item['image'][correct_order[j]-1]
                new_batch.non_tensor_batch['multi_modal_data'][i]['image'] = new_images
                new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = new_answer
            else:
                raise ValueError(f"Unknown question_type: {new_batch.non_tensor_batch['question_type'][i]}")
        elif data_source=='ShowHowTo_e':
            def get_option_img_num(text):
                import re
                cnt = 0
                for match in re.finditer(r'<image>', text.split('Options:')[-1]):
                    cnt += 1
                return cnt
            if new_batch.non_tensor_batch['question_type'][i] in ['description_to_frame', 'missing_frame', 'next_frame']:
                option_img_num = get_option_img_num(new_batch.non_tensor_batch['extra_info'][i]['question'])
                assert option_img_num == 4, f"Unsupported image number: {option_img_num}, ShowHowTo data should have 4 images in options"
                new_batch.non_tensor_batch['multi_modal_data'][i]['image'][-4:], shuffled_idx = randomly_swap_images(
                    item['image'][-4:],
                    seed=seed
                )   # only swap images in choices
                if shuffled_idx!= list(range(len(item['image'][-4:]))):
                    assert new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B', 'C', 'D', 'E'], f"Unsupported ground_truth: {new_batch.non_tensor_batch['reward_model'][i]['ground_truth']}"
                    if new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B', 'C', 'D']:
                        origin_ans_idx = ['A', 'B', 'C', 'D'].index(new_batch.non_tensor_batch["reward_model"][i]["ground_truth"])
                        new_ans_idx = shuffled_idx.index(origin_ans_idx)
                        new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = ['A', 'B', 'C', 'D'][new_ans_idx]
            elif new_batch.non_tensor_batch['question_type'][i] == 'ordering':
                option_img_num = get_option_img_num(new_batch.non_tensor_batch['extra_info'][i]['question'])
                assert option_img_num == 0, f"Unsupported image number: {option_img_num}, ShowHowTo ordering data should have 0 images in options"
                assert new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] in ['A', 'B', 'C', 'D', 'E'], f"Unsupported ground_truth: {new_batch.non_tensor_batch['reward_model'][i]['ground_truth']}"
                # extract correct order
                import re
                def extract_options(text):
                    options = {}
                    lines = text.split('\n')
                    capture = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('Options:'):
                            capture = True
                            continue
                        if capture:
                            if not stripped:
                                break
                            match = re.match(r'^([A-D])\.\s*(.*)$', stripped)
                            if match:
                                key = match.group(1)
                                sequence = list(map(int, match.group(2).split('→')))
                                options[key] = sequence
                    return options
                if new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] != 'E':    # do not swap images for E cause we don't know the correct order
                    options = extract_options(new_batch.non_tensor_batch['extra_info'][i]['question'])
                    correct_order = options[new_batch.non_tensor_batch["reward_model"][i]["ground_truth"]]
                    new_answer = random.choice(['A', 'B', 'C', 'D', 'E'])
                    if new_answer in ['A', 'B', 'C', 'D']:
                        new_order = options[new_answer]
                        new_images = deepcopy(item['image'])
                        for j in range(len(correct_order)):
                            new_images[new_order[j]-1] = item['image'][correct_order[j]-1]
                    elif new_answer == 'E':
                        new_order = list(range(1, 5))
                        random.shuffle(new_order)
                        while new_order in options.values():
                            random.shuffle(new_order)
                        new_images = deepcopy(item['image'])
                        for j in range(len(correct_order)):
                            new_images[new_order[j]-1] = item['image'][correct_order[j]-1]
                    new_batch.non_tensor_batch['multi_modal_data'][i]['image'] = new_images
                    new_batch.non_tensor_batch["reward_model"][i]["ground_truth"] = new_answer
            else:
                raise ValueError(f"Unknown question_type: {new_batch.non_tensor_batch['question_type'][i]}")
        else:
            raise ValueError(f"Unknown data_source: {data_source}")
        # img_num = len(new_batch.non_tensor_batch["multi_modal_data"][i]['image'])
        # from copy import deepcopy
        # imgs = deepcopy(new_batch.non_tensor_batch["multi_modal_data"][i]['image'])
        # # randomly swap range(0, n)
        # shuffled_list = list(range(img_num))
        # random.shuffle(shuffled_list)
        # for j in range(img_num):
        #     imgs[j] = new_batch.non_tensor_batch["multi_modal_data"][i]['image'][shuffled_list[j]]
        # new_batch.non_tensor_batch["multi_modal_data"][i]['image'] = imgs
        new_images = new_batch.non_tensor_batch['multi_modal_data'][i]['image']
        raw_prompt = processor.apply_chat_template([new_batch.non_tensor_batch['raw_prompt'][i][0]], add_generation_prompt=True, tokenize=False)
        model_inputs = processor(text=[raw_prompt], images=new_images, return_tensors="pt", max_pixels=config.get("max_pixels", 100352))
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        new_batch.non_tensor_batch["multi_modal_inputs"][i] = dict(model_inputs)
        # import pdb; pdb.set_trace()
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.max_prompt_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            left_pad=True,
            truncation=config.truncation,
        )
        if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
            print(model_inputs.get("image_grid_thw"))
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        new_batch.batch['input_ids'][i] = input_ids[0]
        new_batch.batch['attention_mask'][i] = attention_mask[0]
        new_batch.batch['position_ids'][i] = position_ids[0]
        # row_dict["input_ids"] = input_ids[0]
        # row_dict["attention_mask"] = attention_mask[0]
        # row_dict["position_ids"] = position_ids[0]
    return new_batch, decay