import cv2
import os

def clahe(img_path, output_path):

    img = cv2.imread(img_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # if img is None:
    #     return False
    # CLAHE

    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe_obj.apply(l)

    lab_clahe = cv2.merge((cl1, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Unsharp masking
    gaussian = cv2.GaussianBlur(img_clahe, (0,0), 1.2)
    sharpened_img = cv2.addWeighted(img_clahe, 1.15, gaussian, -0.15, 0)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, sharpened_img)
    
    return True

def prep_images(data_path, output_path):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_path, split)
        for category in os.listdir(split_dir):
            category_dir = os.path.join(split_dir, category)

            if not os.path.isdir(category_dir):
                continue

            os.makedirs(os.path.join(output_path, split, category), exist_ok=True)

            for img_file in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_file)

                if not os.path.isfile(img_path):
                    continue
                out_path = os.path.join(output_path, split, category, img_file)
                clahe(img_path, out_path)

if __name__ == "__main__":
    data_path = "data"
    output_path = "preprocessed_data"
    prep_images(data_path, output_path)
