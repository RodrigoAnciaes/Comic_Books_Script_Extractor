import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def angle_between_vectors(v1,v2):
    dot_product = np.dot(v1,v2) 
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product/magnitude
    angle = np.arccos(np.clip(cos_angle,-1,1))
    return np.degrees(angle)

def bubble_seg_eq(img):

    img=cv.imread(img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    limit1 = (0, 0, 200)
    limit2 = (180,20, 255)
    mask = cv.inRange(hsv, limit1, limit2)

    mask=cv.dilate(mask, None, iterations=7)
    mask=cv.erode(mask, None, iterations=7)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    M = cv.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    epsilon = 0.001 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
    target_point = (approx[:,0,0].min() + approx[:,0,0].max()) // 2, approx[:,0,1].max()  # Adjust for your specific point of interest

    min_angle = float("inf")
    num_points = len(approx)
    min_angle_point = None
    for i in range(num_points):
        p1 = approx[i][0]
        p2 = approx[(i+1)%num_points][0]
        p3 = approx[(i+2)%num_points][0]
    #cv.circle(img,p2,5,(255,0,0),10)
        v1 = p1 - p2
        v2 = p3 - p2
        angle = angle_between_vectors(v1, v2)
    #print(angle)
        if angle < min_angle and angle > 0 and angle < 90:
            min_angle = angle
            min_angle_point = p2
    if min_angle_point is None:
        print("No suitable point found")
        return None
    direction = min_angle_point - (cx,cy)
    return direction


if __name__ == "__main__":
    image_path = "bubble_11.png"
    direction = bubble_seg_eq(image_path)
    print(direction)
