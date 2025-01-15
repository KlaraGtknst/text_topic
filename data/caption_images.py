import os
from glob import glob
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


# Example usage
if __name__ == '__main__':

    # Load the processor and model
    # `use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version.
    # Legacy behavior is being used. The current behavior will be deprecated in version 5.0.0. In the new behavior, if both images and text are provided, the last token (EOS token) of the input_ids and attention_mask tensors will be removed. To test the new behavior, set `legacy=False`as a processor call argument.
    processor = AutoProcessor.from_pretrained("microsoft/git-base", use_fast=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    # Load and preprocess the image
    l_img_paths = sorted(glob(os.path.join('/Users/klara/Downloads/', "*.png")))
    for image_path in tqdm(l_img_paths):
        print("start image", image_path)

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Generate captions
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("Generated Caption:", caption)
