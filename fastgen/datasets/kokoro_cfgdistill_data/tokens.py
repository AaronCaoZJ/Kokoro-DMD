# Verbatim port of `pil_img2rgb` and `add_special_tokens` from data/data_utils.py of the reference
# CFG-distillation post-training code. The rest of that module is the packed-training path, which
# FastGen has separately (fastgen/networks/kokoro/data_utils.py). Kept byte-identical so the special
# token IDs / vocab match the CFG-distill checkpoint exactly.
import sys  # noqa: F401  (kept for parity with reference module surface)

from PIL import Image


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer, add_kokoro_tokens=False, split_diff_story=False):
    print(f"Initial tokenizer vocabulary size: {len(tokenizer)}")

    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if "<|im_start|>" not in all_special_tokens:
        new_tokens.append("<|im_start|>")

    if "<|im_end|>" not in all_special_tokens:
        new_tokens.append("<|im_end|>")

    if "<|vision_start|>" not in all_special_tokens:
        new_tokens.append("<|vision_start|>")

    if "<|vision_end|>" not in all_special_tokens:
        new_tokens.append("<|vision_end|>")

    # Conditionally add kokoro tokens based on checkpoint compatibility
    if add_kokoro_tokens:
        if "kokoro_cut_continue" not in all_special_tokens:
            new_tokens.append("kokoro_cut_continue")
            print("Adding 'kokoro_cut_continue' token")

        if "kokoro_cut_shift" not in all_special_tokens:
            new_tokens.append("kokoro_cut_shift")
            print("Adding 'kokoro_cut_shift' token")
    else:
        print("Skipping kokoro tokens - not present in checkpoint")
    if split_diff_story:
        if "<|story_mode|>" not in all_special_tokens:
            new_tokens.append("<|story_mode|>")
            print("Adding '<|story_mode|>' token for story splitting")
        if "<|diff_mode|>" not in all_special_tokens:
            new_tokens.append("<|diff_mode|>")
            print("Adding '<|diff_mode|>' token for story splitting")

    num_new_tokens = tokenizer.add_tokens(new_tokens)

    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    # Only get kokoro token IDs if they were added
    if add_kokoro_tokens:
        cut_continue_token_id = tokenizer.convert_tokens_to_ids("kokoro_cut_continue")
        cut_shift_token_id = tokenizer.convert_tokens_to_ids("kokoro_cut_shift")
        print(f"kokoro_cut_continue token ID: {cut_continue_token_id}")
        print(f"kokoro_cut_shift token ID: {cut_shift_token_id}")
    else:
        # Use None or -1 as placeholder values
        cut_continue_token_id = None
        cut_shift_token_id = None
    if split_diff_story:
        story_mode_token_id = tokenizer.convert_tokens_to_ids("<|story_mode|>")
        diff_mode_token_id = tokenizer.convert_tokens_to_ids("<|diff_mode|>")
        print(f"story_mode token ID: {story_mode_token_id}")
        print(f"diff_mode token ID: {diff_mode_token_id}")
    else:
        story_mode_token_id = None
        diff_mode_token_id = None

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
        cut_continue_token_id=cut_continue_token_id,
        cut_shift_token_id=cut_shift_token_id,
        story_mode_token_id=story_mode_token_id,
        diff_mode_token_id=diff_mode_token_id,
    )

    return tokenizer, new_token_ids, num_new_tokens
