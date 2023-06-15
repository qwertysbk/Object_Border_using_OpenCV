import cv2
import rembg

import warnings

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")


class Object_Border:
    def __init__(self, img):
        """pick the image from the local directory

        Args:
            img (_type_): image file path
        """
        self.img = cv2.imread(img)
        self.img = cv2.resize(self.img, (750, 500), interpolation=cv2.INTER_CUBIC)
        self.original = self.img.copy()

    def regionOfInterest(self):
        self.dim = cv2.selectROI("Window to Select", self.img)
        roi = self.img[self.dim[1]:self.dim[1] + self.dim[3], self.dim[0]:self.dim[0] + self.dim[2]]
        return roi

    def removeBG(self, roi):
        """Remove the image background from the Region of Interest

        Args:
            roi (image): The selected region of interest from Image

        Returns:
            image: removed image
        """
        rem = rembg.remove(roi)
        return rem

    def outline(self, removed_image):
        """Automatic Object Online using Contours

        Args:
            removed_image (image): use the removed image to add outline on original image
        """
        gray = cv2.cvtColor(removed_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.img[self.dim[1]:self.dim[1] + self.dim[3], self.dim[0]:self.dim[0] + self.dim[2]],
                         contours,
                         -1,
                         (0, 255, 0),
                         2)
        cv2.imwrite('output_img.jpg', self.img)
        cv2.imshow("output", self.img)

    def displayImages(self, removed_bg):
        cv2.imshow("RemBG", removed_bg)
        cv2.imwrite('Output_Image.jpg', self.img)


def main():
    # choose image file
    testImage = 'Image.jpeg'
    # call the Class instance
    assessment = Object_Border(testImage)
    # select the region of interest
    roi = assessment.regionOfInterest()
    # remove the background from region of interest
    removed_bg = assessment.removeBG(roi)
    # add outline on the original image
    assessment.outline(removed_bg)
    # display all the images
    assessment.displayImages(removed_bg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()