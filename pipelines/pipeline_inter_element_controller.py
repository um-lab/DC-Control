from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
import torch.nn.functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module

from pipelines.pipeline_intra_element_controller import (
    ControlNetModel_Union,
    MultiControlNetUnionModel,
    StableDiffusionXLControlNetDotPipeline,
    StableDiffusionXLPipelineOutput,
    retrieve_timesteps,
)


class StableDiffusionXLControlNetInterElementControllerPipeline(StableDiffusionXLControlNetDotPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        condition_prompt: Union[str, List[str]] = None,
        image_list: List[PipelineImageInput] = None,
        control_type=None,
        layout_image=None,
        layout_type=None,
        control_conditions: Optional[List[Any]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        if control_conditions is not None:
            if not isinstance(control_conditions, (list, tuple)) or len(control_conditions) != 3:
                raise ValueError(
                    "`control_conditions` must be a list/tuple of length 3: [layout, control_type, layout_type]"
                )
            layout_image, control_type, layout_type = control_conditions

        is_multi_element = (
            isinstance(image_list, (list, tuple))
            and len(image_list) > 0
            and isinstance(image_list[0], (list, tuple))
        )
        if not is_multi_element:
            return super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                condition_prompt=condition_prompt,
                image_list=image_list,
                control_type=control_type,
                layout_image=layout_image,
                layout_type=layout_type,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                denoising_end=denoising_end,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                target_size=target_size,
                negative_original_size=negative_original_size,
                negative_crops_coords_top_left=negative_crops_coords_top_left,
                negative_target_size=negative_target_size,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            from diffusers.utils import deprecate

            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            from diffusers.utils import deprecate

            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetUnionModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetUnionModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        do_classifier_free_guidance = guidance_scale > 1.0

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel_Union)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                do_classifier_free_guidance,
            )

        if isinstance(image_list, (list, tuple)) and len(image_list) == 0:
            raise ValueError("`image_list` must be a non-empty list for multi-element inference.")
        num_elements = len(image_list)

        if condition_prompt is None:
            condition_prompts = ["" for _ in range(num_elements)]
        elif isinstance(condition_prompt, str):
            condition_prompts = [condition_prompt for _ in range(num_elements)]
        elif isinstance(condition_prompt, (list, tuple)) and len(condition_prompt) == num_elements:
            condition_prompts = list(condition_prompt)
        else:
            raise ValueError("`condition_prompt` must be a string or a list with the same length as `image_list`.")

        if not isinstance(layout_image, (list, tuple)) or len(layout_image) != num_elements:
            raise ValueError(
                "For multi-element inference, `layout` must be a list of images with the same length as `image_list`."
            )

        def _to_one_hot_list(x, length: int, label: str) -> List[torch.Tensor]:
            if x is None:
                raise ValueError(f"`{label}` must be provided for multi-element inference.")
            if isinstance(x, torch.Tensor):
                if x.dim() == 1 and x.numel() == length:
                    return [x.float() for _ in range(num_elements)]
                if x.dim() == 2 and x.shape[0] == num_elements and x.shape[1] == length:
                    return [x[i].float() for i in range(num_elements)]
                if x.dim() == 1 and x.numel() == num_elements:
                    return [F.one_hot(x[i].long(), num_classes=length).float() for i in range(num_elements)]
                raise ValueError(f"Unsupported tensor shape for `{label}`: {tuple(x.shape)}")
            if isinstance(x, (list, tuple)):
                if len(x) != num_elements:
                    raise ValueError(f"`{label}` length must match number of elements ({num_elements}).")
                out = []
                for v in x:
                    if isinstance(v, torch.Tensor) and v.numel() == length:
                        out.append(v.float())
                    else:
                        out.append(F.one_hot(torch.tensor(int(v), dtype=torch.long), num_classes=length).float())
                return out
            return [
                F.one_hot(torch.tensor(int(x), dtype=torch.long), num_classes=length).float() for _ in range(num_elements)
            ]

        control_type_list = _to_one_hot_list(control_type, 8, "control_type")
        layout_type_list = _to_one_hot_list(layout_type, 4, "layout_type")

        for elem_idx in range(num_elements):
            elem_images = image_list[elem_idx]
            if not isinstance(elem_images, (list, tuple)) or len(elem_images) != 8:
                raise ValueError("For multi-element inference, each `image_list[i]` must be a list of length 8.")
            for img_idx in range(len(elem_images)):
                if isinstance(elem_images[img_idx], PIL.Image.Image) or isinstance(elem_images[img_idx], torch.Tensor):
                    image_prepared = self.prepare_image(
                        image=elem_images[img_idx],
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    height, width = image_prepared.shape[-2:]
                    elem_images[img_idx] = image_prepared
            image_list[elem_idx] = list(elem_images)

        layout_tensors = []
        for elem_idx in range(num_elements):
            layout_tensor = self.prepare_image(
                image=layout_image[elem_idx],
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            layout_tensors.append(layout_tensor)
        dot_hidden_states = torch.stack(layout_tensors, dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel_Union) else keeps)

        if isinstance(image_list, list) and len(image_list) > 0 and isinstance(image_list[0], (list, tuple)):
            original_size = original_size or layout_tensors[0].shape[-2:]
        else:
            original_size = original_size or dot_hidden_states.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        layout_control_type_tensor = torch.stack(layout_type_list, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        if do_classifier_free_guidance:
            layout_control_type_tensor = torch.cat([layout_control_type_tensor, layout_control_type_tensor], dim=0)
        layout_control_type_tensor = (
            layout_control_type_tensor.to(device=device, dtype=prompt_embeds.dtype)
            .repeat(num_images_per_prompt, 1, 1)
        )

        condition_prompt_embeds_list = []
        condition_pooled_embeds_list = []
        control_type_batch_list = []
        per_control_type = [ct.to(device=device, dtype=prompt_embeds.dtype) for ct in control_type_list]
        control_batch_size = batch_size * num_images_per_prompt * (2 if do_classifier_free_guidance else 1)

        for elem_idx in range(num_elements):
            (
                cond_prompt_embeds,
                neg_cond_prompt_embeds,
                pooled_cond_prompt_embeds,
                neg_pooled_cond_prompt_embeds,
            ) = self.encode_prompt(
                condition_prompts[elem_idx],
                None,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                None,
                None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            add_condition_text_embeds = pooled_cond_prompt_embeds
            if do_classifier_free_guidance:
                cond_prompt_embeds = torch.cat([neg_cond_prompt_embeds, cond_prompt_embeds], dim=0)
                add_condition_text_embeds = torch.cat([neg_pooled_cond_prompt_embeds, add_condition_text_embeds], dim=0)

            condition_prompt_embeds_list.append(cond_prompt_embeds.to(device))
            condition_pooled_embeds_list.append(add_condition_text_embeds.to(device))
            control_type_batch_list.append(per_control_type[elem_idx].reshape(1, -1).repeat(control_batch_size, 1))

        def _get_controlnet_conditioning_scale(step_idx: int) -> Union[float, List[float]]:
            if isinstance(controlnet_conditioning_scale, list):
                if len(controlnet_conditioning_scale) == 0:
                    raise ValueError("`controlnet_conditioning_scale` must not be empty.")
                if len(controlnet_conditioning_scale) == num_elements:
                    return controlnet_conditioning_scale
                return controlnet_conditioning_scale
            return controlnet_conditioning_scale

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                per_element_down = []
                per_element_mid = []

                cond_scale = _get_controlnet_conditioning_scale(i)

                for elem_idx in range(num_elements):
                    controlnet_added_cond_kwargs = {
                        "text_embeds": condition_pooled_embeds_list[elem_idx],
                        "time_ids": add_time_ids,
                        "control_type": control_type_batch_list[elem_idx],
                    }

                    elem_scale = (
                        cond_scale[elem_idx] if isinstance(cond_scale, list) and len(cond_scale) == num_elements else cond_scale
                    )
                    if isinstance(controlnet_keep[i], list):
                        elem_scale = [c * s for c, s in zip(elem_scale, controlnet_keep[i])]
                    else:
                        if isinstance(elem_scale, list):
                            elem_scale = elem_scale[0]
                        elem_scale = elem_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=condition_prompt_embeds_list[elem_idx],
                        controlnet_cond_list=image_list[elem_idx],
                        conditioning_scale=elem_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )

                    per_element_down.append(down_block_res_samples)
                    per_element_mid.append(mid_block_res_sample)

                num_layers = len(per_element_down[0])
                down_block_res_samples_by_layer = [
                    torch.stack([per_element_down[e][layer_idx] for e in range(num_elements)], dim=0)
                    for layer_idx in range(num_layers)
                ]
                mid_block_res_samples_by_element = torch.stack(per_element_mid, dim=0)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    layout_control_type=layout_control_type_tensor,
                    dot_hidden_states=dot_hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples_by_layer,
                    mid_block_additional_residual=mid_block_res_samples_by_element,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and callback_steps is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
