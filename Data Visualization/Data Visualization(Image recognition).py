#Data Visualization(Image recognition)

import matplotlib.pyplot as plt
import cv2

def visualize_detection(frame, detections, h, w):
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

…                          (0, 0, 255), 2)  # Draw the bounding box
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)  # Add text label

    # Show the output image using Matplotlib
    plt.imshow(frame)
    plt.axis('off')  # Hide axes
    plt.show()
