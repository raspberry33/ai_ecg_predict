import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim



def load_and_preprocess_image(image_path, size=(300, 300)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return img



def compare_histograms(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity



def compare_ssim(image1, image2):
    similarity, _ = ssim(image1, image2, full=True)
    return similarity



def is_ekg_image(reference_image_path, test_image_path):
    ref_img = load_and_preprocess_image(reference_image_path)
    test_img = load_and_preprocess_image(test_image_path)

    hist_similarity = compare_histograms(ref_img, test_img)
    ssim_similarity = compare_ssim(ref_img, test_img)


   
    avg_similarity = (hist_similarity + ssim_similarity) / 2 * 100
    return avg_similarity


if __name__ == '__main__':

	reference_image_path = 'images/Screenshot-2018-10-27-09.09.14.png'
	test_image_path = 'images/wishlist_board.png'
	similarity_percentage = is_ekg_image(reference_image_path, test_image_path)
	print(f"Відсоток подібності з ЕКГ: {similarity_percentage:.2f}%")

	if similarity_percentage > 21:
	    print("+")
else:
	    print("-")
	    
