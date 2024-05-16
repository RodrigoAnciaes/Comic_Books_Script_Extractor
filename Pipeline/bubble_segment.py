import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def line_coefficients(point1, point2):
    # Convert points to numpy array of shape (N,1,2)
        points = np.array([point1, point2], dtype=np.float32).reshape(-1, 1, 2)
    
    # Fit a line to the points
        vx, vy, x, y = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
    
    # Calculate the slope (m) and y-intercept (b) of the line
        m = vy / vx
        b = y - m * x
    
        return m, b

def angle_between_vectors(v1,v2):
    dot_product = np.dot(v1,v2) 
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product/magnitude
    angle = np.arccos(np.clip(cos_angle,-1,1))
    return np.degrees(angle)

def bubble_seg_eq(img):

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
    closest_point = min(approx, key=lambda point: np.linalg.norm(point[0] - target_point))

    min_angle = float("inf")
    num_points = len(approx)
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




    m,b = line_coefficients((cx, cy), (min_angle_point[0], min_angle_point[1])) # Line equation coefficients
    return m,b


# usage:
