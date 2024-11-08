import os
import cv2

# File Paths
filedir = os.getcwd()  # Use os.getcwd() instead of __file__

# Get User Name
name = input("Enter Name: ")

# Create directory
directory = os.path.join(filedir, "dataset", name)
if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory {} Successfully Created".format(name))
else:
    print("Directory Already Exists. Continuing With Capture")

# Capture Images
print("Starting Webcam...")
capture = cv2.VideoCapture(0)
image_counter = 1

while True:
    _, frame = capture.read()
    cv2_imshow(frame) # Use cv2_imshow instead of cv2.imshow
    k = cv2.waitKey(100) & 0xff
    if k == 27:  # ESC pressed
        print("Escape hit. Closing Webcam...")
        break
    elif k == 32:  # SPACE pressed
        print("Writing file")
        image_name = "opencv_frame_{}.png".format(image_counter)
        cv2.imwrite(os.path.join(directory, image_name), frame)
        print("{} written!".format(image_name))
        image_counter += 1

capture.release()
cv2.destroyAllWindows()