import cv2
import numpy as np

class Image:
    def __init__(self):
        self.path = None
        self.data = None
        # self.load_image()

    def reset(self):
        self.data = None

    def load_image(self, path):
        self.path = path
        if not isinstance(path, str) or not path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            raise ValueError("invalid image source: unsupported file type")
        elif not cv2.haveImageReader(path):
            raise ValueError("invalid image source: cannot read the file")
        else:
            self.data = cv2.imread(path)
        
    def resize_image(self, target_rows=256, target_cols=256):
        resized = cv2.resize(self.data, (target_cols, target_rows))
        return resized
    
    def crop_image(self, width, height):
        if self.data is None:
            raise ValueError("invalid image")
        
        h, w = self.data.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        x1 = max(0, center_x - width // 2)
        y1 = max(0, center_y - height // 2)
        x2 = min(w, x1 + width)
        y2 = min(h, y1 + height)

        cropped = self.data[y1:y2, x1:x2]
        return cropped
    
    def convert_to_grayscale(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        return self.data

    def quality_check(self, threshold=100):
        if self.data is None:
            raise ValueError("invalid image")
        laplacian_var = cv2.Laplacian(self.data, cv2.CV_64F).var()
        return laplacian_var >= threshold # lower laplacian variance means more blurry image

    def normalize_image(self):
        self.data = cv2.normalize(self.data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return self.data
    
    def save_image(self, output_path):
        cv2.imwrite(output_path, self.data)

if __name__ == "__main__":
    img = Image()
    img.load_image("../lib/test_images/img0.jpeg")
    new_img = img.resize_image(256, 256)
    column, row = new_img.data.shape[:2]
    print(f"resized image size: {column} x {row}")

    # img.crop_image(100, 100)
    # img.save_image("../lib/outputs/img1_cropped.jpg")