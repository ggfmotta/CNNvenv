import matplotlib.pyplot as plt
import cv2 # version 4.2.0

# print(cv2.__version__)

def rotate_img(img, case):
    """
    rotates image according to case:
    case = 1 -> 90° clockwise rotation
    case = 2 -> 90° counterclockwise rotation
    case = 3 -> 180° rotation 
    """
    if case == 1:
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
    elif case == 2:
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    else:
        img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    
    return img_rotated


# # re# read image
# img = cv2.imread('PotatoEaters.jpg')

# # Select rotation case - see rotate_img function
# case = 2

# img2 = rotate_img(img, case)

# # Save image
# cv2.imwrite('imgrotated.jpg',img2)

# img = cv2.imread('PotatoEaters.jpg')

# # Select rotation case - see rotate_img function
# case = 2

# img2 = rotate_img(img, case)

# # Save image
# cv2.imwrite('imgrotated.jpg',img2)

# # Debug options
# plot = True

# if plot:
#     fig, ax = plt.subplots(2,1)
#     ax[0].imshow(img)
#     ax[1].imshow(img2)

#     plt.show()
