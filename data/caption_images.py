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

    def __call__(self, image_path: str) -> str:
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

    # TODO: save results to a file


if __name__ == '__main__':

    # Load the processor and model
    captioner = ImageCaptioner()

    # Load and preprocess the image
    l_img_paths = sorted(glob(os.path.join('/Users/klara/Downloads/', "*.png")))
    for image_path in tqdm(l_img_paths):
        print("start image", image_path)
        caption = captioner(image_path)
        print("Generated Caption:", caption)
