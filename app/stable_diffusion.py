import os

from PIL import Image
from keras_cv.models import StableDiffusionV2


class Leo:
    model = StableDiffusionV2(
        img_height=512,
        img_width=768,
        jit_compile=True,
    )

    def __call__(self, image_id: str, prompt: str, epochs: int):
        render, *_ = self.model.text_to_image(prompt=prompt, num_steps=epochs)
        img = Image.fromarray(render)
        img.save(os.path.join("app", "images", f"{image_id}.png"))


if __name__ == '__main__':
    leo = Leo()
    leo(
        image_id="001",
        prompt="High resolution 3d model. Leopard sitting in front of a computer wearing a green hoodie.",
        epochs=20,
    )
