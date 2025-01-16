from utils.os_manipulation import exists_or_create
import os
from glob import glob
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


class ImageCaptioner:

    def __init__(self):
        """
        Initialize the image captioning model.
        The model is based on the `microsoft/git-base` model.
        Find supplementary information about the model here:
        https://huggingface.co/docs/transformers/main/tasks/image_captioning (15.01.2025)
        """
        # `use_fast` = `True`: fast version, as soon as available
        #       (not yet, 15.01.2025: falling back to the slow version)
        # Legacy behavior is being used: if both images and text are provided,
        #       the last token (EOS token) of the input_ids and attention_mask tensors will be removed.
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base",
                                                       use_fast=True, legacy=False)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    def caption_image(self, image_path: str) -> str:
        """
        Generate a caption for the given image.
        :param image_path: A path to the image file including the file name and extension.
        :return: The generated caption for the image.
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # Generate captions
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def save_caption_to_file(self, image_path: str, save_path: str):
        """
        Save the generated caption to a file.
        :param image_path: A path to the image file including the file name and extension.
        :param save_path: Path to save the caption file excluding the file name and extension.
        """
        caption = self.caption_image(image_path)
        image_name = '_'.join(image_path.split('/')[-1].split('.')[:len(image_path.split('/')[-1].split('.')) - 1])
        exists_or_create(path=save_path)
        with open(os.path.join(save_path, f"{image_name}_caption.txt"), "w") as file:
            file.write(caption)
        # print('saved caption to file:', os.path.join(save_path, f"{image_name}_caption.txt"))


if __name__ == '__main__':

    # Load the processor and model
    captioner = ImageCaptioner()

    # Load and preprocess the image
    l_img_paths = sorted(glob(os.path.join('/Users/klara/Downloads/', "*.png")))
    for image_path in tqdm(l_img_paths):
        # print("start image", image_path)
        caption = captioner.caption_image(image_path)
        captioner.save_caption_to_file(image_path, save_path="/Users/klara/Downloads/captions/")
        # print("Generated Caption:", caption)
