import abc
import argparse
import numpy as np
import os
import random
import torch
import torch.nn.functional as nnf
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Dict

import ptp_utils
import seq_aligner
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from sentence_template import SENTENCE_TEMPLATE

SCENE_TYPE = ["urban mainroad"]
OBJECTS = ['car',
           'van',
           'truck',
           'buses', ]
LIGHTING_CONDITION = ["bright", "dim"]
TIME_AND_WEATHER_CONDITIONS = ["sunny day", "clear night", "cloudy day", "rainy night"]


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                        1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,
                         image_name=None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), image_name=image_name)


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0, image_name=None):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), image_name=image_name)


# %%
def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, image_name=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                                  num_inference_steps=NUM_DIFFUSION_STEPS,
                                                  guidance_scale=GUIDANCE_SCALE, generator=generator,
                                                  low_resource=LOW_RESOURCE)

    ptp_utils.view_images(images, image_name=image_name)
    return images, x_t


def run(prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                                  num_inference_steps=NUM_DIFFUSION_STEPS,
                                                  guidance_scale=GUIDANCE_SCALE, generator=generator,
                                                  low_resource=LOW_RESOURCE)

    return images, x_t


def sentence_generation(label_dict):
    # randomly select one element from the sentence template
    sentence_template = SENTENCE_TEMPLATE[torch.randint(0, len(SENTENCE_TEMPLATE), (1,)).item()]
    for key in label_dict:
        assert key in sentence_template, f"{key} not in {sentence_template}"
        sentence_template = sentence_template.replace(key, label_dict[key])
    return sentence_template


if __name__ == '__main__':
    LOW_RESOURCE = False
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    CHECKPOINT = 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000
    CHECKPOINT = [f"checkpoint-{checkpoint}" for checkpoint in CHECKPOINT]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/home/turing/cfs_cz/finn/codes/diffusers'
                                '/examples/text_to_image/experiments/full_training/'
                                'jidu-data-II-15-v06/bs_12_lr_1e5_2gpu_200_epoch')
    parser.add_argument('--save_dir', type=str,
                        default='/home/turing/cfs_cz/finn/codes/prompt-to-prompt/visualization/jidu-data-II-sensor-v06/bs_12_lr_1e5_2gpu_200_epoch')
    parser.add_argument('--img_name', nargs='+', default="cross_self_exp", type=str, help='image name for generation')
    parser.add_argument("--method", type=str, default="replace", choices=["replace", "refine", "reweight"])
    parser.add_argument("--num_images", type=int, default=5, help="number of images to generate")
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    for checkpoint in CHECKPOINT:
        if checkpoint != None:
            unet = UNet2DConditionModel.from_pretrained(os.path.join(args.model_path, checkpoint, "unet"),
                                                        use_safetensors=True)
            ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_path, unet=unet, use_safetensors=True).to(
                device)
            save_dir = os.path.join(args.save_dir, checkpoint)
            os.makedirs(save_dir, exist_ok=True)
        else:
            ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_path, use_safetensors=True).to(device)
        tokenizer = ldm_stable.tokenizer
        cross_replace_steps_list = [0.2, 0.4, 0.6, 0.8, 0.9]
        self_replace_steps_list = [0.2, 0.4, 0.6, 0.8, 0.9]
        cross_replace_steps_list = [0.8]
        self_replace_steps_list = [0.4]
        # self_replace_steps_list = [0.5 for i in range(4)]
        # cross_replace_steps_list = [0.5 for i in range(4)]
        random.seed(42)
        if len(args.img_name) == 1:
            args.img_name = args.img_name[0]
        g_cpu = torch.Generator()
        for i in tqdm(range(args.num_images)):
            sentence_dict = {}
            sentence_dict["[scene type]"] = SCENE_TYPE[0]
            sentence_dict["[time and weather conditions]"] = TIME_AND_WEATHER_CONDITIONS[0]
            sentence_dict["[lighting condition]"] = LIGHTING_CONDITION[0]

            sentence_dict[
                "[objects in the image and their quantity]"] = (f"{random.randint(1, 5)} car,"
                                                                f"{random.randint(1, 5)} van,")
            # sentence_dict[
            #     "[objects in the image and their quantity]"] = "several cars"
            for exp in ['caption', 'template']:
                if exp == "caption":
                    prompts_original = (f"{SCENE_TYPE[0]},{TIME_AND_WEATHER_CONDITIONS[0]},"
                                        f"{random.randint(1, 5)} car,"
                                        f"{random.randint(1, 5)} van,"
                                        f"{LIGHTING_CONDITION[0]}")
                else:
                    prompts_original = sentence_generation(sentence_dict)

                # prompts_original = "The urban mainroad is a medley of 5 cars,1 van, under bright during a sunny day."
                prompts_edited = prompts_original.replace("sunny day", "clear night")
                prompts = [prompts_original, prompts_edited]
                print(f"prompts: {prompts}")
                controller = AttentionStore()
                image, x_t = run_and_display([prompts_original], controller, latent=None, run_baseline=False,
                                             generator=g_cpu,
                                             image_name=None)
                for cross_replace_step, self_replace_step in zip(cross_replace_steps_list, self_replace_steps_list):
                    if args.method == "replace":
                        controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=cross_replace_step,
                                                      self_replace_steps=self_replace_step)
                        _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False,
                                            image_name=os.path.join(save_dir,
                                                                    f"{exp}_{i}_replace_cross_{cross_replace_step}_self_{self_replace_step}_p2p"))
                    elif args.method == "refine":
                        controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                                     self_replace_steps=0.8)
                        _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False,
                                            image_name=os.path.join(save_dir, f"{args.img_name}_refine_p2p"))
                    else:
                        raise NotImplementedError
