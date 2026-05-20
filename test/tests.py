import pytest
import cv2
import numpy as np
import sys
import os

CLASSES_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLASSES_FOLDER)

from utils.utils_image import Image 
from utils.utils_image import ImageStack

# test the image class
class TestImage:
    
    @pytest.fixture
    def img(self):
        img = Image()
        return img

    def test_load_image_success(self, img):
        img.load_image("../lib/test_images/img0.jpeg")
        assert img.image is not None, "test image loaded"

    def test_load_image_failure(self, img):
        with pytest.raises(FileNotFoundError):
            img.load_image("../lib/test_images/img2.jpeg")

    def test_resize_image(self, img):
        img.load_image("lib/test_images/img0.jpeg")
        img.resize_image((100, 100))
        assert img.image.size == (100, 100), "resized successfully"

    def test_crop_image(self, img):
        img.load_image("../lib/test_images/img0.jpeg")
        img.crop_image((150, 50))
        assert img.image.size == (40, 40), "cropped successfully"

    def test_save_image(self, img):
        img.load_image("../lib/test_images/img0.jpeg")
        img.crop_image(256, 256)
        img.save_image("../lib/outputs/img0_cropped.jpg")
        assert os.path.exists("../lib/outputs/img0_cropped.jpg"), "modified image saved"

class TestImageStack:
    @pytest.fixture
    def img_stack(self):
        img_stack = ImageStack()
        return img_stack

    def test_create_image_stack(self, img_stack):
        img_stack.create_image_stack("lib/test_images")
        assert len(img_stack.get_images()) == 1, "created image stack"

    def test_resize_images(self, img_stack):
        img_stack.create_image_stack("lib/outputs")
        img_stack.resize_images()
        for img in img_stack.get_images():
            assert img.image.size == (256, 256), "resized image stack"

if __name__ == "__main__":
    pytest.main()
