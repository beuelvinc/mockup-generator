import cv2
import numpy as np

class MockupGenerator:
    def __init__(self, frame_path, photo_path, inset_ratio=0.05):
        """
        Initializes the MockupGenerator with the frame and photo paths.

        Parameters:
        frame_path (str): Path to the frame template image.
        photo_path (str): Path to the photo to be inserted.
        inset_ratio (float): The ratio to inset the photo from the frame's edges to preserve borders. Default is 0.05 (5%).
        """
        self.frame = cv2.imread(frame_path)
        self.photo = cv2.imread(photo_path)
        self.inset_ratio = inset_ratio
        self.ordered_frame_corners = None
        self.inset_corners = None

    def detect_frame(self):
        """
        Detects the frame's position in the template image and orders the corners.
        
        This method identifies the largest contour in the frame image, approximates it to a quadrilateral, 
        and orders the corners for consistent mapping.
        """
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(frame_contour, True)
        approx = cv2.approxPolyDP(frame_contour, epsilon, True)
        
        if len(approx) == 4:
            frame_corners = np.array([point[0] for point in approx], dtype='float32')
            self._sort_corners(frame_corners)
        else:
            raise ValueError("Frame detection failed: Expected a quadrilateral.")
    
    def _sort_corners(self, frame_corners):
        """
        Sorts the detected frame corners into a consistent order (top-left, top-right, bottom-right, bottom-left).

        Parameters:
        frame_corners (ndarray): Array of detected frame corners.
        """
        sums = frame_corners.sum(axis=1)
        diffs = np.diff(frame_corners, axis=1)
        self.ordered_frame_corners = np.array([
            frame_corners[np.argmin(sums)],  
            frame_corners[np.argmin(diffs)],
            frame_corners[np.argmax(sums)], 
            frame_corners[np.argmax(diffs)] 
        ], dtype='float32')
        self._calculate_inset_corners()
    
    def _calculate_inset_corners(self):
        """
        Calculates the inset corners for the photo placement inside the frame.

        The inset corners are slightly smaller than the actual frame corners to leave the frame's borders intact.
        """
        self.inset_corners = self.ordered_frame_corners * (1 - self.inset_ratio) + \
                             np.mean(self.ordered_frame_corners, axis=0) * self.inset_ratio
    
    def warp_photo(self):
        """
        Warps the photo to fit inside the detected frame area.

        Returns:
        ndarray: The warped photo image.
        """

        h, w = self.photo.shape[:2]
        photo_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(photo_corners, self.inset_corners)
        return cv2.warpPerspective(self.photo, M, (self.frame.shape[1], self.frame.shape[0]))
    
    def combine_images(self, warped_photo):
        """
        Combines the warped photo with the frame, preserving the frame's borders.

        Parameters:
        warped_photo (ndarray): The warped photo image.

        Returns:
        ndarray: The final combined image with the photo inside the frame.
        """
        mask = np.zeros_like(self.frame, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(self.inset_corners)], (255, 255, 255))
        frame_mask = cv2.bitwise_not(mask)
        frame_with_photo_area = cv2.bitwise_and(self.frame, frame_mask)
        return cv2.add(frame_with_photo_area, cv2.bitwise_and(warped_photo, mask))
    
    def create_mockup(self, output_path=None):
        """
        Creates the final mockup by detecting the frame, warping the photo, and combining them.

        Parameters:
        output_path (str, optional): Path to save the final mockup image. If not provided, the image is not saved.

        Returns:
        ndarray: The final mockup image with the photo inside the frame.
        """
        self.detect_frame()
        warped_photo = self.warp_photo()
        final_image = self.combine_images(warped_photo)
        
        if output_path:
            cv2.imwrite(output_path, final_image)
        return final_image
    

mockup = MockupGenerator('./images/templates/template2.jpg', './images/image.jpg')

final_image = mockup.create_mockup('./images/outputs/mockup_from_solution2.png')
