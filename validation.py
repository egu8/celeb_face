from train_utils import evaluate_pic

if __name__ =="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Parser for Validation images')
    parser.add_argument('--path_to_img', type=str, default="test_image.jpg",
                        help='path to validation image')

    args = parser.parse_args()

    # Evaluate on one image
    from PIL import Image
    img = Image.open(args.path_to_img).convert("RGB")
    evaluate_pic(img)