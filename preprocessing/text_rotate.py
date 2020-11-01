# import the necessary packages
import numpy as np
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image_body", required=True, help="path to input image_body file")
# args = vars(ap.parse_args())
# load the image_body from disk
# image_body = cv2.imread(r"D:\123.png")

def img_rotate(image):
    # convert the image_body to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image_body, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image_body to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)


    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # imgs = (image_body, gray, thresh, rotated)

    # for i in range(len(imgs)):
    #     cv2.imwrite(rf"D:\{i}.png", imgs[i])

    return rotated

# draw the correction angle on the image_body so we can validate it
# cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# show the output image_body
# print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("Input", image_body)
# cv2.imshow("Gray", gray)
# cv2.imshow("Thresh", thresh)
# cv2.imshow("Rotated", rotated)
# cv2.waitKey(0)
