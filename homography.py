import numpy as np
import cv2
import sys

# Random choice for ransac_for_homography
def generate_random_set(src_key_points, dst_key_points):
    r = np.random.choice(len(src_key_points), 4)
    return np.array([src_key_points[i] for i in r]), np.array([dst_key_points[i] for i in r])

# Calculate homograpy matrix
def cal_H(src, dest):
    A = []
    # Transform four (x,y) coordinates to calculate homography matrix.
    for i in range(len(src)):
        x, y = src[i][0], src[i][1]
        x_prime, y_prime = dest[i][0], dest[i][1]
        A.append([x, y, 1, 0, 0, 0, -x * x_prime, -x_prime * y, -x_prime])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y, -y_prime])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)

    H = np.reshape(V[-1], (3, 3)) # Use the last column vector of the SVD result V as H
    H = (1 / H.item(8)) * H
    return H

# Calculate distance for ransac 
def dist(src, dest, H):
    # x_prime is H*x
    x = np.transpose(np.matrix([src[0], src[1], 1]))
    x_prime = np.transpose(np.matrix([dest[0], dest[1], 1]))
    
    # x_hat is estimation result.
    x_hat = np.dot(H, x)
    x_hat = x_hat/x_hat.item(2)
    
    e = x_prime - x_hat
    return np.linalg.norm(e)

# Calculate homography matrix by ransac
def ransac_for_homography(src_key_points, dst_key_points, good_src_key_points, good_dst_key_points, threshold = 150, iter_limit = 4000):
    max_inlier_cnt = 0
    max_H = 0
    iter_cnt = 0
    
    while max_inlier_cnt < threshold and iter_cnt < iter_limit:
        iter_cnt += 1
        src, dest = generate_random_set(good_src_key_points, good_dst_key_points)
        H = cal_H(src, dest)
        
        inlier_cnt = 0
        for j in range(len(src_key_points)):
            d = dist(src_key_points[j], dst_key_points[j], H)
            if d < 5:
                inlier_cnt += 1
            
            if max_inlier_cnt < inlier_cnt :
                max_inlier_cnt = inlier_cnt
                max_H = H
    return max_H

# forward mapping
def forward_mapping(img_left, img_right, H):
    # [[x ...]
    #  [y ...]
    #  [1 ...]]
    src_locs = []
    for x in range(img_right.shape[1]):
        for y in range(img_right.shape[0]):
            loc = [x, y, 1]
            src_locs.append(loc)
    src_locs = np.array(src_locs).transpose()
    
    # [[x ...]     [[x' ...]     [[x' ...]
    #  [y ...]  ->  [y' ...]  ->  [y' ...]]
    #  [1 ...]]     [1  ...]] 
    dst_locs = np.matmul(H, src_locs)
    dst_locs = dst_locs / dst_locs[2, :]
    dst_locs = dst_locs[:2, :]
    src_locs = src_locs[:2, :]
    dst_locs = np.round(dst_locs, 0).astype(int)
    
    height, width, _ = img_left.shape
    result = np.zeros((height, width * 2, 3), dtype = int) # prepare a panorama image
    for src, dst in zip(src_locs.transpose(), dst_locs.transpose()):
        if dst[0] < 0 or dst[1] < 0 or dst[0] >= width*2 or dst[1] >= height:
            continue
        
        result[dst[1], dst[0]] = img_right[src[1], src[0]]
    result[0: height, 0 : width] = img_left
    return result

# forward_mapping with interpolation
def forward_mapping_interpolation_median(img_left, img_right, H):
    # [[x ...]
    #  [y ...]
    #  [1 ...]]
    src_locs = []
    for x in range(img_right.shape[1]):
        for y in range(img_right.shape[0]):
            loc = [x, y, 1]
            src_locs.append(loc)
    src_locs = np.array(src_locs).transpose()
    
    # [[x ...]     [[x' ...]     [[x' ...]
    #  [y ...]  ->  [y' ...]  ->  [y' ...]]
    #  [1 ...]]     [1  ...]] 
    dst_locs = np.matmul(H, src_locs)
    dst_locs = dst_locs / dst_locs[2, :]
    dst_locs = dst_locs[:2, :]
    src_locs = src_locs[:2, :]
    dst_locs = np.round(dst_locs, 0).astype(int)
    
    height, width, _ = img_left.shape
    result = np.zeros((height, width * 2, 3), dtype = int) # prepare a panorama image
    for src, dst in zip(src_locs.transpose(), dst_locs.transpose()):
        if dst[0] < 0 or dst[1] < 0 or dst[0] >= width*2 or dst[1] >= height:
            continue
        
        result[dst[1], dst[0]] = img_right[src[1], src[0]]
    result[0: height, 0 : width] = img_left
    
    # Interpolation by median value
    interpolation_locs = [] # Need to interpolation pixels
    for x in range(width * 2):
        for y in range(height):
            if np.sum(result[y, x] == [0,0,0]) == 3:
                loc = [x, y, 1]
                interpolation_locs.append(loc)
    
    interpolation_locs = np.array(interpolation_locs).transpose()
    
    H_inverse = np.linalg.inv(H)
    ori_locs = np.matmul(H_inverse, interpolation_locs) # Original pixels index
    ori_locs = ori_locs / ori_locs[2, :]
    ori_locs = ori_locs[:2, :]
    interpolation_locs = interpolation_locs[:2, :]
    ori_locs = np.round(ori_locs, 0).astype(int)
    
    for ori, res in zip(ori_locs.transpose(), interpolation_locs.transpose()):
        if ori[1] >= height or ori[0] >= width or ori[0] < 0 or ori[1] < 0:
            continue
        
        # Find the original rgb value nearby(3x3) to perform interpolation
        near_rgb = [img_right[ori[1], ori[0]]]
        up_plag, down_plag, left_plag, right_flag = False, False, False, False
        if ori[1] > 0:
            near_rgb.append(img_right[ori[1]-1, ori[0]])
            up_plag = True
        if ori[1] < height-1:
            near_rgb.append(img_right[ori[1]+1, ori[0]])
            down_plag = True
        if ori[0] > 0:
            near_rgb.append(img_right[ori[1], ori[0]-1])
            left_plag = True
        if ori[0] < width-1:
            near_rgb.append(img_right[ori[1], ori[0]+1])
            right_flag = True
        if up_plag and left_plag:
            near_rgb.append(img_right[ori[1]-1, ori[0]-1])
        if up_plag and right_flag:
            near_rgb.append(img_right[ori[1]-1, ori[0]+1])
        if down_plag and left_plag:
            near_rgb.append(img_right[ori[1]+1, ori[0]-1])
        if down_plag and right_flag:
            near_rgb.append(img_right[ori[1]+1, ori[0]+1])
        
        near_rgb = np.array(near_rgb)
        median_b = np.median(near_rgb.transpose()[0])
        median_g = np.median(near_rgb.transpose()[1])
        median_r = np.median(near_rgb.transpose()[2])
        result[res[1], res[0]] = [median_b, median_g, median_r]
    
    return result

# backward_mapping
def backward_mapping(img_left, img_right, H):
    dst_locs = []
    for x in range(img_right.shape[1], img_right.shape[1] * 2):
        for y in range(img_right.shape[0]):
            loc = [x, y, 1]
            dst_locs.append(loc)
    dst_locs = np.array(dst_locs).transpose()
    
    H_inverse = np.linalg.inv(H)
    
    src_locs = np.matmul(H_inverse, dst_locs)
    src_locs = src_locs / src_locs[2, :]
    src_locs = src_locs[:2, :]
    src_locs = np.round(src_locs, 0).astype(int)
    
    dst_locs = dst_locs[:2, :]
    height, width, _ = img_right.shape
    result = np.zeros((height, width * 2, 3), dtype = int)
    for src, dst in zip(src_locs.transpose(), dst_locs.transpose()):
        if src[1] >= height or src[0] >= width or src[0] < 0 or src[1] < 0:
            continue
        
        result[dst[1], dst[0]] = img_right[src[1], src[0]]
    result[0: height, 0 : width] = img_left
    return result

def main():
    # 1. choose two images
    img_left_path = sys.argv[1]
    img_right_path = sys.argv[2]
    assert len(sys.argv) == 3, 'invalid arguments\nYou should input two arguments, left_image_path and right_image_path. \nYou must enter as \'python homography.py [left_image_path] [right_image_path]\''
    
    img_left = cv2.imread(img_left_path) 
    img_right = cv2.imread(img_right_path)
    
    img_left_gray = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
    
    # 2. compute ORB key_primeoint and descriptors
    orb = cv2.ORB_create()
    key_point_left, descriptors_left = orb.detectAndCompute(img_left_gray, None)
    key_point_right, descriptors_right = orb.detectAndCompute(img_right_gray, None)
    
    # 3. apply Bruteforce matching with Hamming distance (opencv)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors_left, descriptors_right)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:100]

    dst_key_points = np.float32([key_point_left[match.queryIdx].pt for match in matches]).reshape((-1, 2))
    src_key_points = np.float32([key_point_right[match.trainIdx].pt for match in matches]).reshape((-1, 2))

    good_dst_key_points = np.float32([key_point_left[match.queryIdx].pt for match in good_matches]).reshape((-1, 2))
    good_src_key_points = np.float32([key_point_right[match.trainIdx].pt for match in good_matches]).reshape((-1, 2)) 
    
    # 4. implement RANSAC algorithm to compute the homography matrix. (DIY)
    H = ransac_for_homography(src_key_points, dst_key_points, good_src_key_points, good_dst_key_points)
    
    # 5. prepare a panorama image of larger size (DIY)
    # 6. warp two images to the panorama image using the homography matrix (DIY)
    forward_result = forward_mapping(img_left, img_right, H)
    forward_result_median = forward_mapping_interpolation_median(img_left, img_right, H)
    backward_result = backward_mapping(img_left, img_right, H)
    
    cv2.imwrite('./result_forward.png', forward_result)
    cv2.imwrite('./result_forward_interpolation_median.png', forward_result_median)
    cv2.imwrite('./result_backward.png', backward_result)

if __name__ == "__main__":
    main()