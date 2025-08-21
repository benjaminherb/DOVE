from pathlib import Path
import argparse
import logging
from datetime import datetime
import time
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

from transformers import set_seed
from typing import Dict, Tuple
from diffusers.models.embeddings import get_3d_rotary_pos_embed

import traceback
import json
import os
import cv2
from PIL import Image

from pathlib import Path
import pyiqa
import imageio.v3 as iio
import glob

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
log = logging.getLogger(__name__)


def setup_logging(log_filename, log_level=logging.INFO, log_folder="logs"):
    script_dir = Path(__file__).parent
    log_dir = os.path.join(script_dir, log_folder)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in video_exts)


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path):
    # return a tensor of shape [F, C, H, W] // 0, 1
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")


def save_frames_as_png(video, output_dir, start_id=0, fps=8):
    """
    Save video frames as PNG sequence.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_dir (str): directory to save PNG files
        fps (int): kept for API compatibility
    """
    video = video[0]  # Remove batch dimension
    video = video.permute(1, 2, 3, 0)  # [F, H, W, C]

    os.makedirs(output_dir, exist_ok=True)
    frames = (video.cpu() * 255).clamp(0, 255).to(torch.uint8).numpy()

    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"{i+start_id:03d}.png")
        Image.fromarray(frame).save(filename)


def preprocess_video_match(
    video_path: Path | str,
    is_match: bool = False,
    overlap_t = 0
) -> torch.Tensor:
    """
    Loads a single video.

    Args:
        video_path: Path to the video file.
    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    
    # prepend frames to avoid "ghosting" at the start
    frames = torch.cat([frames[1:overlap_t//2+1].flip(0), frames], dim=0)

    F, H, W, C = frames.shape
    original_shape = (F, H, W, C)

    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = F % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        #pad_h = (16 - H % 16) % 16
        #pad_w = (16 - W % 16) % 16
        pad_h = (48 - H % 48) % 48
        pad_w = (48 - W % 48) % 48
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding(video, pad_f=0, pad_h=0, pad_w=0):
    if pad_f > 0:
        video = video[:, :, :-pad_f, :, :]
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]

    return video


def make_temporal_chunks(F, chunk_len, overlap_t=8):
    """
    Args:
        F: total number of frames
        chunk_len: int, chunk length in time (excluding overlap)
        overlap: int, number of overlapping frames between chunks
    Returns:
        time_chunks: List of (start_t, end_t) tuples
    """

    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t - 1, effective_stride))
    #if chunk_starts[-1] + chunk_len < F:
        #chunk_starts.append(F - chunk_len)

    time_chunks = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    log.info(f"Time Chunks ({len(time_chunks)}): {time_chunks}")
    return time_chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    """
    Args:
        H, W: height and width of the frame
        tile_size_hw: Tuple (tile_height, tile_width)
        overlap_hw: Tuple (overlap_height, overlap_width)
    Returns:
        spatial_tiles: List of (start_h, end_h, start_w, end_w) tuples
    """
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    h_tiles = list(range(0, H, tile_height))
    w_tiles = list(range(0, W, tile_width))


    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    spatial_tiles = []
    for ht in h_tiles:
        h_start = max(ht - overlap_h, 0)
        h_end = min(ht + tile_width + overlap_h, H)
        for wt in w_tiles:
            w_start = max(wt - overlap_w, 0)
            w_end = min(wt + tile_width + overlap_w, W)
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    log.info(f"Spatial Tiles: {spatial_tiles}")

    return spatial_tiles


def make_spatial_tiles_split(H, W, split_hw, overlap_hw=(32, 32)):
    """
    Create spatial tiles by splitting image into specified number of pieces.

    Args:
        H: Height of the image
        W: Width of the image
        split_hw: Tuple of (h_splits, w_splits) - number of pieces to split height and width into
        overlap_hw: Tuple of (overlap_h, overlap_w) - overlap added on top of calculated tile sizes

    Returns:
        List of tuples (h_start, h_end, w_start, w_end) defining each tile
    """
    h_splits, w_splits = split_hw
    overlap_h, overlap_w = overlap_hw

    if h_splits == 0 or w_splits == 0:
        return [(0, H, 0, W)]

    if h_splits == 1 and w_splits == 1:
        return [(0, H, 0, W)]

    tile_height = np.ceil(H / h_splits).astype(int)
    tile_width = np.ceil(W / w_splits).astype(int)

    spatial_tiles = []
    for current_h in np.arange(0, H, tile_height):

        h_start = np.max([0, current_h - overlap_h])
        h_end = np.min([H, current_h + tile_height + overlap_h])

        for current_w in np.arange(0, W, tile_width):
            w_start = np.max([0, current_w - overlap_w])
            w_end = np.min([W, current_w + tile_width + overlap_w])

            spatial_tiles.append((h_start, h_end, w_start, w_end))

    log.info(f"Spatial Tiles: {spatial_tiles}")
    return spatial_tiles

def get_valid_chunk_region(t_start, t_end, video_shape, overlap_t):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    out_t_start = np.max([t_start + valid_t_start - overlap_t // 2, 0])
    out_t_end = t_start + valid_t_end - overlap_t // 2

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
    }

def get_valid_tile_region(h_start, h_end, w_start, w_end,
                          video_shape, overlap_h, overlap_w, blend_width=0):
    _, _, F, H, W = video_shape


    if blend_width * 2 > overlap_h or blend_width * 2 > overlap_w:
        log.warning(f"blend_width {blend_width} is larger than {overlap_h} or {overlap_w}, setting blend_width to overlap // 2")
        blend_width = min([overlap_h // 2, overlap_w // 2])

    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_h_start = 0 if h_start == 0 else overlap_h - blend_width//2
    valid_h_end = h_len if h_end == H else h_len - overlap_h + blend_width//2
    valid_w_start = 0 if w_start == 0 else overlap_w - blend_width//2
    valid_w_end = w_len if w_end == W else w_len - overlap_w + blend_width//2

    out_h_start = h_start + valid_h_start
    out_w_start = w_start + valid_w_start
    # Ensure output region matches valid region size
    out_h_end = out_h_start + (valid_h_end - valid_h_start)
    out_w_end = out_w_start + (valid_w_end - valid_w_start)

    return {
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Dict,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    if transformer_config.patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (
            num_frames + transformer_config.patch_size_t - 1
        ) // transformer_config.patch_size_t
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )

    return freqs_cos, freqs_sin


def create_blend_mask(blend_size, tile_region, video_height, video_width):
    height = tile_region["out_h_end"] - tile_region["out_h_start"]
    width = tile_region["out_w_end"] - tile_region["out_w_start"]
    mask = torch.ones(height, width)

    if blend_size == 0:
        return mask

    if tile_region["out_h_start"] > 0: # not top
        for i in range(blend_size):
            mask[i, :] *= (i+.5) / blend_size

    if tile_region["out_h_end"] < video_height: # not bottom
        for i in range(blend_size):
            mask[height - 1 - i, :] *= (i+.5) / blend_size

    if tile_region["out_w_start"] > 0: # not left
        for i in range(blend_size):
            mask[:, i] *= (i+.5) / blend_size

    if tile_region["out_w_end"] < video_width: # not right
        for i in range(blend_size):
            mask[:, width - 1 - i] *= (i+.5) / blend_size
    return mask

@no_grad
def process_video(
    pipe: CogVideoXPipeline,
    video: torch.Tensor,
    prompt: str = '',
    noise_step: int = 0,
    sr_noise_step: int = 399,
):
    # SR the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.

    # Decode video
    log.debug("Decode Video")
    video = video.to(pipe.vae.device, dtype=pipe.vae.dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    latent = latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is not None:
        ncopy = latent.shape[2] % patch_size_t
        # Copy the first frame ncopy times to match patch_size_t
        first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
        latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)

        assert latent.shape[2] % patch_size_t == 0

    batch_size, num_channels, num_frames, height, width = latent.shape

    # Get prompt embeddings
    log.debug("Get prompt embeddings")
    prompt_token_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_token_ids = prompt_token_ids.input_ids
    prompt_embedding = pipe.text_encoder(
        prompt_token_ids.to(latent.device)
    )[0]
    _, seq_len, _ = prompt_embedding.shape
    prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

    latent = latent.permute(0, 2, 1, 3, 4)

    # Add noise to latent (Select)
    log.debug("Add Noise")
    if noise_step != 0:
        noise = torch.randn_like(latent)
        add_timesteps = torch.full(
            (batch_size,),
            fill_value=noise_step,
            dtype=torch.long,
            device=latent.device,
        )
        latent = pipe.scheduler.add_noise(latent, noise, add_timesteps)

    timesteps = torch.full(
        (batch_size,),
        fill_value=sr_noise_step,
        dtype=torch.long,
        device=latent.device,
    )

    # Prepare rotary embeds
    log.debug("Prepare rotary embeds")
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    transformer_config = pipe.transformer.config
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=transformer_config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=latent.device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # Free Up Memory (Testing)
    del video, latent_dist, prompt_token_ids
    torch.cuda.empty_cache()

    # Predict noise
    log.debug("Predict Noise")
    predicted_noise = pipe.transformer(
        hidden_states=latent,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        return_dict=False,
    )[0]

    # Free Up Memory (Testing)
    del prompt_embedding, rotary_emb
    torch.cuda.empty_cache()

    log.debug("Get Velocity")
    latent_generate = pipe.scheduler.get_velocity(
        predicted_noise, latent, timesteps
    )

    # Free Up Memory (Testing)
    del predicted_noise, latent


    # generate video
    log.debug("Generate Video")
    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, ncopy:, :, :, :]

    # [B, C, F, H, W]
    log.debug("Output Preparation")
    video_generate = pipe.decode_latents(latent_generate)
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)

    log.debug("Output")
    return video_generate



def main(args):
    # Setup logging
    settings_string = f"DOVE_{args.upscale}x_T{args.chunk_len}_{args.overlap_t}_" \
                      f"S{args.tile_split_hw[0] if args.tile_split_hw[0] != 0 else args.tile_size_hw[0]}_{args.overlap_hw[0]}_B{args.tile_blend}"
    setup_logging(log_filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{settings_string}.log", log_level=logging.INFO)

    log.info("Starting DOVE Inference Script")
    log.info("Arguments:\n" + '\n'.join([f'  {k}: {v}' for k,v in args.__dict__.items()]))

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Invalid dtype. Choose from 'float16', 'bfloat16', or 'float32'.")

    if args.chunk_len > 0:
        log.info(f"Chunking video into {args.chunk_len} frames with {args.overlap_t} overlap")
        overlap_t = args.overlap_t
    else:
        overlap_t = 0
    if args.tile_size_hw != (0, 0):
        log.info(f"Tiling video into {args.tile_size_hw} frames with {args.overlap_hw} overlap")
        overlap_hw = args.overlap_hw
    elif  args.tile_split_hw != (0,0):
        log.info(f"Splitting video into {args.tile_split_hw} tiles with {args.overlap_hw} overlap")
        overlap_hw = args.overlap_hw
    else:
        overlap_hw = (0, 0)

    # Set seed
    set_seed(args.seed)

    torch.cuda.memory._record_memory_history(max_entries=100000)
    os.makedirs(args.output_path, exist_ok=True)

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    if args.lora_path:
        log.info(f"Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(
            args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0) # lora_scale = lora_alpha / rank

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if args.is_cpu_offload:
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # Process Video
    video_path = args.input_path
    saved_frames = 0

    video_name = os.path.basename(video_path)
    output_name = video_name.rsplit('.', 1)[0] + "." + settings_string
    output_dir = os.path.join(args.output_path, output_name)
    metadata_file = os.path.join(args.output_path, output_name + '.json')
    
    print(metadata_file)
    if os.path.exists(metadata_file):
        log.info(f"{metadata_file} already exists! Skipping!")
        return

    prompt = ""

    video_start_time = time.time()
    log.info(f"Starting {video_name} ({args.upscale}x)")

    # Read video
    # [F, C, H, W]
    video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(video_path, is_match=True, overlap_t=args.overlap_t)
    H_, W_ = video.shape[2], video.shape[3]
    video = torch.nn.functional.interpolate(video, size=(H_*args.upscale, W_*args.upscale), mode=args.upscale_mode, align_corners=False)
    __frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]) # -1, 1
    video = torch.stack([__frame_transform(f) for f in video], dim=0)

    video = video.unsqueeze(0)
    # [B, C, F, H, W]
    video = video.permute(0, 2, 1, 3, 4).contiguous()

    _B, _C, _F, _H, _W = video.shape
    time_chunks = make_temporal_chunks(_F, args.chunk_len, overlap_t)

    if args.tile_split_hw[0] != 0 and args.tile_split_hw[1] != 0:
        spatial_tiles = make_spatial_tiles_split(_H, _W, args.tile_split_hw, overlap_hw)
    else:
        spatial_tiles = make_spatial_tiles(_H, _W, args.tile_size_hw, overlap_hw)

    # Source Res
    log.info(f"Process video: {video_name} | Prompt: {prompt} | Frame: {_F} (ori: {original_shape[0]}; pad: {pad_f}) | Source Resolution {H_}, {W_} | Target Resolution: {_H}, {_W} (ori: {original_shape[1]*args.upscale}, {original_shape[2]*args.upscale}; pad: {pad_h}, {pad_w}) | Chunk Num: {len(time_chunks)*len(spatial_tiles)}")

    for time_chunk_id, (t_start, t_end) in enumerate(time_chunks):

        chunk_start_time = time.time()


        chunk_region = get_valid_chunk_region(
            t_start, t_end, video_shape=video.shape, overlap_t=overlap_t)

        # Skip if the first frame of the chunk already exists
        first_frame = os.path.join(output_dir, f"{chunk_region['out_t_start']:03d}.png")
        last_frame = os.path.join(output_dir, f"{chunk_region['out_t_end']:03d}.png")
        if os.path.exists(first_frame) and os.path.exists(last_frame):
            log.info(f"Skipping time chunk {time_chunk_id} ({t_start}-{t_end}) as {os.path.basename(first_frame)} and {os.path.basename(last_frame)} already exists!")
            continue
        log.info(f"Starting time chunk ({time_chunk_id+1}/{len(time_chunks)}) - Frame {t_start}-{t_end}")
        max_chunk_size = 0
        video_generate = torch.zeros_like(video[:, :, chunk_region["valid_t_start"]:chunk_region["valid_t_end"] , :, :])

        for spatial_tile_id, (h_start, h_end, w_start, w_end) in enumerate(spatial_tiles):

            video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            max_chunk_size = max([max_chunk_size, video_chunk.numel()])


            torch.cuda.empty_cache()
            log.info(f"Processing chunk ({spatial_tile_id+1}/{len(spatial_tiles)}): {list(video_chunk.shape)} | t: {t_start}:{t_end} | h: {h_start}:{h_end} | w: {w_start}:{w_end}")
            log.info("Before")
            log.info(f"H {h_start}-{h_end} / W {w_start}-{w_end}")
            log.info(video_chunk.shape)

            # [B, C, F, H, W]
            try:
                # Log input and output shape for process_video for debugging
                log.info(f"process_video input shape: {video_chunk.shape}")
                _video_generate = process_video(
                    pipe=pipe,
                    video=video_chunk,
                    prompt=prompt,
                    noise_step=args.noise_step,
                    sr_noise_step=args.sr_noise_step,
                )
                log.info(f"process_video output shape: {_video_generate.shape}")
            except Exception as e:
                log.error(f"Exception: {e}\n{traceback.format_exc()}")
                torch.cuda.memory._dump_snapshot(f"./exceptions/{datetime.now().strftime('%Y%m%d_%H%M%S')}_EXCEPTION_{video_name}_memory_snapshot_{t_start}_{h_start}_{w_start}.pickle")
                return

            tile_region = get_valid_tile_region(
                h_start, h_end, w_start, w_end,
                video_shape=video.shape,
                overlap_h=overlap_hw[0],
                overlap_w=overlap_hw[1],
                blend_width=args.tile_blend
            )

            mask = create_blend_mask(args.tile_blend, tile_region, video_height=_H, video_width=_W).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]

            # DEBUG
            log.info(f"Debug info for tile {spatial_tile_id+1}:")
            log.info(f"_video_generate shape: {_video_generate.shape}")
            log.info(f"video_generate shape: {video_generate.shape}")
            log.info(f"tile_region: {tile_region}")
            log.info(f"chunk_region: {chunk_region}")
            log.info(f"Mask shape: {mask.shape}")
            left_tensor_shape = video_generate[:,:,:,tile_region["out_h_start"]:tile_region["out_h_end"],
                                                     tile_region["out_w_start"]:tile_region["out_w_end"]].shape
            log.info(f"Left tensor shape: {left_tensor_shape}")
            right_tensor_premask = _video_generate[:, :, chunk_region["valid_t_start"]:chunk_region["valid_t_end"],
                                                        tile_region["valid_h_start"]:tile_region["valid_h_end"],
                                                        tile_region["valid_w_start"]:tile_region["valid_w_end"]].shape
            log.info(f"Right tensorpremask shape: {right_tensor_premask}")
            right_tensor_shape = (_video_generate[:, :, chunk_region["valid_t_start"]:chunk_region["valid_t_end"],
                                                        tile_region["valid_h_start"]:tile_region["valid_h_end"],
                                                        tile_region["valid_w_start"]:tile_region["valid_w_end"]].cpu() * mask).shape
            log.info(f"Right tensor shape: {right_tensor_shape}")


            video_generate[:,:,:,tile_region["out_h_start"]:tile_region["out_h_end"],
                                 tile_region["out_w_start"]:tile_region["out_w_end"]] += \
                _video_generate[:, :, chunk_region["valid_t_start"]:chunk_region["valid_t_end"],
                                      tile_region["valid_h_start"]:tile_region["valid_h_end"],
                                      tile_region["valid_w_start"]:tile_region["valid_w_end"]].cpu() * mask

        torch.cuda.empty_cache()

        log.info("Removing padding")
        video_generate = remove_padding(video_generate, pad_h=pad_h*args.upscale, pad_w=pad_w*args.upscale)
        if time_chunk_id+1 == len(time_chunks): # only remove time padding for last chunk
            video_generate = remove_padding(video_generate, pad_f=pad_f)
        if time_chunk_id == 0: # remove prepended frames
            video_generate = video_generate[:, :,overlap_t//2:, :,:]

        # Save as PNG sequence
        log.info(f"Saving video time chunk {time_chunk_id+1} ({chunk_region['out_t_start']}-{chunk_region['out_t_end']}) to {output_dir}")
        save_frames_as_png(video_generate, output_dir, chunk_region['out_t_start'], fps=args.fps)
        chunk_elapsed_time = time.time() - chunk_start_time
        spf_theoretical = chunk_elapsed_time / (t_end - t_start)
        spf_real = chunk_elapsed_time / (chunk_region['out_t_end'] - chunk_region['out_t_start'])
        saved_frames += chunk_region['out_t_end'] - chunk_region['out_t_start']
        log.info(f"Processed video chunk {time_chunk_id+1} in {chunk_elapsed_time:.0f} seconds with {spf_real:.0f} spf ({spf_theoretical:.0f}) with a max chunk of {max_chunk_size}")


    video_end_time = time.time()
    video_elapsed_time = video_end_time - video_start_time
    spf = 0 if saved_frames == 0 else video_elapsed_time / saved_frames
    log.info(f"Finished processing {video_name} with {saved_frames} frames in {video_elapsed_time:.0f} seconds with {spf:.0f} spf")

    metadata = {
        "start_time": datetime.fromtimestamp(video_start_time).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(video_end_time).strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": str(video_elapsed_time),
        "spf": str(spf),
        "source": os.path.basename(video_path),
        "source_path": video_path,
        "destination": output_dir,
        "arguments": args.__dict__
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSR using DOVE")

    parser.add_argument("--input_path", "-i", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--output_path", "-o", type=str, default="./results", help="The path save generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--upscale_mode", type=str, default="bilinear")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)
    parser.add_argument("--is_cpu_offload", action="store_true", help="Enable CPU offload for the model")
    parser.add_argument("--is_vae_st", action="store_true", help="Enable VAE slicing and tiling")
    # Crop and Tiling Parameters
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0), help="Tile size for spatial tiling (height, width)")
    parser.add_argument("--tile_split_hw", type=int, nargs=2, default=(0, 0), help="Tile split size for spatial tiling (height, width)")
    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--tile_blend", type=int, default=0, help="Overlap for blending tiles")
    parser.add_argument("--chunk_len", type=int, default=0, help="Chunk length for temporal chunking")
    parser.add_argument("--overlap_t", type=int, default=8)

    args = parser.parse_args()

    main(args)
