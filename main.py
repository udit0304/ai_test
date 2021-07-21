import os.path
import cv2
from skimage.metrics import structural_similarity as ssim


# function to measure similarity with labelled images and test images.
def comapre_similarirty(test_img, imgs, label):
    sift = cv2.xfeatures2d.SIFT_create()
    test_img = cv2.imread(test_img, 0)
    test_img = cv2.resize(test_img, (512, 512))
    test_kp, test_desc = sift.detectAndCompute(test_img, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for i in range(len(imgs)):
        src_img = cv2.imread(imgs[i], 0)
        src_img = cv2.resize(src_img, (512, 512))
        src_kp, src_desc = sift.detectAndCompute(src_img, None)
        matches = flann.knnMatch(src_desc, test_desc, k=2)
        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        struc_sim = ssim(src_img, test_img)
        if len(good_points) / len(matches) > 0.5 or struc_sim > 0.7:
            return label[i]
    return None


if __name__ == "__main__":
    imgs = []
    test_imgs = []
    labels = []
    path = "."
    valid_images = [".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(f)
        # creating labels as from the file name
        labels.append(f.split("_")[0])

    # reading all test images from the directory test
    for f in os.listdir(path + "/test"):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        test_imgs.append(path + "/test/"+f)
        labels.append(f.split("_")[0])

    # checking each image with similarity function with labeled images
    for test_img in test_imgs:
        test_label = comapre_similarirty(test_img, imgs, labels)
        if test_label is not None:
            print("{} is {}".format(test_img, test_label))
        else:
            print("{} is unidentified".format(test_img))
