#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import itertools
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from parser_helper import parse_args
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from data_module import KontextDataset, collate_fn
from utils import ( 
    import_model_class_from_model_name_or_path,
    tokenize_prompt,
    encode_prompt,
    save_model_card
)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

args = parse_args()

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two

def log_validation(
    pipeline,
    args,
    accelerator,
    dataloader,
    tag,
    is_final_validation=False,
):
    logger.info(f"Running {tag}... \n ")
    pipeline = pipeline.to(accelerator.device)
    
    # Use appropriate precision context
    if accelerator.mixed_precision == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif accelerator.mixed_precision == "fp16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
    with autocast_ctx:
        images = []
        prompts = []
        control_images = []
        target_images = []
        
        # Process full dataset - no limit on samples for validation
        # This ensures we use all images from the test dataset
        
        for batch in dataloader:
            prompt = batch['prompts']
            control_image = batch['source_image']
            target_image = batch['target_image']
            
            # Process in smaller batches to avoid memory issues
            validation_batch_size = 1
            
            for i in range(0, len(prompt), validation_batch_size):
                batch_prompt = prompt[i:i+validation_batch_size]
                batch_control_image = control_image[i:i+validation_batch_size]
                batch_target_image = target_image[i:i+validation_batch_size]
                
                result = pipeline(
                    prompt=batch_prompt,
                    height=args.height,
                    width=args.width,
                    image=batch_control_image, 
                    num_inference_steps=4,
                    generator=generator,
                    guidance_scale=30,
                ).images
                
                images.extend(result)
                prompts.extend(batch_prompt)
                control_images.extend(batch_control_image)
                target_images.extend(batch_target_image)

    # Log to trackers
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            
            for input_img, gen_img, tgt_img, prompt in zip(control_images, images, target_images, prompts):
                formatted_images.append(wandb.Image(input_img, caption="Source Image"))
                formatted_images.append(wandb.Image(tgt_img, caption="Target Image"))
                formatted_images.append(wandb.Image(gen_img, caption=prompt))
            
            tracker.log({tracker_key: formatted_images})
    del images, prompts, control_images, target_images
    del pipeline
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16

            transformer = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
            )
            pipeline = FluxKontextPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            # sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            # sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            # sample_dataloader = accelerator.prepare(sample_dataloader)
            # pipeline.to(accelerator.device)

            # for example in tqdm(
            #     sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            # ):
            #     images = pipeline(example["prompt"]).images

            #     for i, image in enumerate(images):
            #         hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
            #         image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
            #         image.save(image_filename)

            del pipeline
            free_memory()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    # Dataset and DataLoaders creation:
    train_dataset = KontextDataset(
        dataset_name=args.dataset_name,
        source_column_name=args.source_column,
        target_column_name=args.target_column,
        caption_column_name=args.caption_column,
        size=(args.width, args.height),
        split="train",  # Training split
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    if args.validation_check:
        validation_dataset = KontextDataset(
            dataset_name=args.validation_dataset_name,
            source_column_name=args.source_column,
            target_column_name=args.target_column,
            caption_column_name=args.caption_column,
            size=(args.width, args.height),
            split="test",  # Test split
        )

        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, max_sequence_length=77)
            tokens_two = tokenize_prompt(
                tokenizer_two, args.instance_prompt, max_sequence_length=args.max_sequence_length
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt, max_sequence_length=77)
                class_tokens_two = tokenize_prompt(
                    tokenizer_two, args.class_prompt, max_sequence_length=args.max_sequence_length
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_check is None:
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "SAKS_context_blur_instructive"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                        tokens_two = tokenize_prompt(
                            tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                        )
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts,
                        )
                else:
                    elems_to_repeat = len(prompts)
                    if args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[
                                tokens_one.repeat(elems_to_repeat, 1),
                                tokens_two.repeat(elems_to_repeat, 1),
                            ],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=args.instance_prompt,
                        )

                # Convert images to latent space
                if args.cache_latents:
                    if args.vae_encode_mode == "sample":
                        model_input = latents_cache[step].sample()
                    else:
                        model_input = latents_cache[step].mode()
                else:
                    # pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    pixel_values = batch["target_image"].to(dtype=vae.dtype) # target image
                    source_pixel_values = batch["source_image"].to(dtype=vae.dtype) # source image
                    if args.vae_encode_mode == "sample":
                        model_input = vae.encode(pixel_values).latent_dist.sample() #target image
                        source_model_input = vae.encode(source_pixel_values).latent_dist.sample() #source image
                    else:
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        source_model_input = vae.encode(source_pixel_values).latent_dist.mode()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                # replicating the above code for the source images as well
                source_model_input = (source_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                source_model_input = source_model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # print(f"latent_image_ids shape: {latent_image_ids.shape}")
                ###########
                image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2, accelerator.device,
                    weight_dtype,
                )
                # print(f"image_ids shape: {image_ids.shape}")
                # image ids are the same as latent ids with the first dimension set to 1 instead of 0
                image_ids[..., 0] = 1

                latent_image_ids = torch.cat([latent_image_ids, image_ids], dim=0)  # dim 0 is sequence dimension
                # print(f"latent_image_ids shape after concatenation: {latent_image_ids.shape}")
                ########
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise #adding noise on the target image
                # print(f"noisy_model_input shape: {noisy_model_input.shape}")

                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                #converting the source model input to packed latents which are basically patchfying and flattening the source model input
                packed_source_model_input = FluxKontextPipeline._pack_latents(
                    source_model_input,
                    batch_size=source_model_input.shape[0],
                    num_channels_latents=source_model_input.shape[1],
                    height=source_model_input.shape[2],
                    width=source_model_input.shape[3],
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # concatenate the packed noisy model input with the packed source model input across channels dimensions
                transformer_input = torch.cat(
                    [packed_noisy_model_input, packed_source_model_input], dim=1
                )

                # print(f"packed_noisy_model_input shape: {packed_noisy_model_input.shape}")
                # print(f"packed_source_model_input shape: {packed_source_model_input.shape}")
                # print(f"transformer_input shape: {transformer_input.shape}")
                # print(f"latent_image_ids shape: {latent_image_ids.shape}")
                # print(f"text_ids shape: {text_ids.shape}")
                

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=transformer_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # noise on only the target image
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input


                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
                # Run validation every args.validation_steps steps (e.g., every 5 steps)
                if global_step % args.validation_steps == 0:
                    pipeline = FluxKontextPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),
                        torch_dtype=weight_dtype,
                        vae=vae,
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        text_encoder=text_encoder_one,
                        text_encoder_2=text_encoder_two,
                    )
                    
                    log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        dataloader=validation_dataloader,
                        tag="validation",
                    )

            if global_step >= args.max_train_steps:
                break


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            modules_to_save["text_encoder"] = text_encoder_one
        else:
            text_encoder_lora_layers = None

        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        # Final inference
        # Load previous pipeline
        # free_memory()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # transformer = FluxTransformer2DModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        # )
        # pipeline = FluxKontextPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     transformer=transformer,
        #     revision=args.revision,
        #     variant=args.variant,
        #     torch_dtype=weight_dtype,
        # )
        # # load attention processors
        # pipeline.load_lora_weights(args.output_dir)
        # print("last Loaded lora weights successfully from", args.output_dir)

        # # run inference
        # images = []
        # if args.validation_check and args.num_validation_images > 0:
        #     images = log_validation(
        #             pipeline=pipeline,
        #             args=args,
        #             accelerator=accelerator,
        #             tag="Test",
        #             dataloader=validation_dataloader,
        #             is_final_validation=True
        #         )
        #     del pipeline
        #     free_memory()

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt, #TODO: Fix this
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
