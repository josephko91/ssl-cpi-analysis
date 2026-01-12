import cv2
import pandas as pd
import os
import numpy as np

def get_img(file_path):
    return cv2.imread(file_path)

def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_thresh(gray, thresh_val=50):
    return cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)[1]

def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []
    largest = max(contours, key=cv2.contourArea)
    return largest, contours

def aspect_ratio_rect(cnt):
    rect = cv2.minAreaRect(cnt)
    x, y = rect[1][0], rect[1][1]
    rect_length = max(x, y)
    rect_width = min(x, y)
    return rect_width / rect_length if rect_length > 0 else np.nan

def aspect_ratio_ellipse(cnt):
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        widthE, heightE = ellipse[1][0], ellipse[1][1]
        return min(widthE, heightE) / max(widthE, heightE) if max(widthE, heightE) > 0 else np.nan
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        return min(w, h) / max(w, h) if max(w, h) > 0 else np.nan

def extreme_points_std(cnt):
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return np.std([left, right, top, bottom])

def contour_area(cnt):
    return cv2.contourArea(cnt)

def contour_perimeter(cnt):
    return cv2.arcLength(cnt, True)

def convex_hull(cnt):
    return cv2.convexHull(cnt)

def convex_perimeter(hull):
    return cv2.arcLength(hull, True)

def hull_area(hull):
    return cv2.contourArea(hull)

def min_enclosing_circle(cnt):
    _, radius = cv2.minEnclosingCircle(cnt)
    return radius

def filled_circular_area_ratio(area, radius):
    return area / (np.pi * radius ** 2) if radius > 0 else np.nan

def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def circularity(area, perim):
    return (4.0 * np.pi * area) / (perim ** 2) if perim > 0 else np.nan

def roundness(area, convex_perim):
    return (4.0 * np.pi * area) / (convex_perim ** 2) if convex_perim > 0 else np.nan

def perim_area_ratio(perim, area):
    return perim / area if area > 0 else np.nan

def solidity(area, hull_area_val):
    return area / hull_area_val if hull_area_val > 0 else np.nan

def equiv_d(area):
    return np.sqrt(4 * area / np.pi) if area > 0 else np.nan

def complexity(area, perim, radius):
    Ac = np.pi * radius ** 2 if radius > 0 else np.nan
    if area > 0 and perim > 0 and Ac > 0:
        return 10 * (0.1 - (area / (np.sqrt(area / Ac) * perim ** 2)))
    else:
        return np.nan

def process_img(path):
    try:
        img = get_img(path)
        if img is None:
            raise ValueError("Corrupt or unreadable image")
        gray = get_gray(img)
        thresh = get_thresh(gray)
        cnt, contours = find_largest_contour(thresh)
        if cnt is None:
            raise ValueError("No contour found")
        phi = aspect_ratio_rect(cnt)
        phi_elip = aspect_ratio_ellipse(cnt)
        extreme_pts = extreme_points_std(cnt)
        area = contour_area(cnt)
        perim = contour_perimeter(cnt)
        hull = convex_hull(cnt)
        convex_perim_val = convex_perimeter(hull)
        hull_area_val = hull_area(hull)
        radius = min_enclosing_circle(cnt)
        filled_circ_area_ratio = filled_circular_area_ratio(area, radius)
        lap_var = laplacian_var(gray)
        circ = circularity(area, perim)
        roundn = roundness(area, convex_perim_val)
        perim_area = perim_area_ratio(perim, area)
        solid = solidity(area, hull_area_val)
        equiv = equiv_d(area)
        complx = complexity(area, perim, radius)
        return [
            path, phi, phi_elip, extreme_pts, area, perim, filled_circ_area_ratio,
            complx, circ, roundn, perim_area, solid, equiv, lap_var
        ]
    except Exception as e:
        print(f"Corrupt or failed image: {path} ({e})")
        return [path] + [np.nan]*13

def main():
    filepaths_txtfile = '/home/jko/ssl-cpi-analysis/data/labeled_png_filepaths.txt'
    with open(filepaths_txtfile, 'r') as f:
        png_filepaths = [line.strip() for line in f if line.strip()]
    record_list = []
    N = 100  # Print progress every N files
    for idx, f in enumerate(png_filepaths, 1):
        # Extract classification label (second to last part of the path)
        classification = os.path.basename(os.path.dirname(f))
        record = process_img(f)
        record_with_label = [classification] + record
        record_list.append(record_with_label)
        if idx % N == 0:
            print(f"Processed {idx} / {len(png_filepaths)} files")
    columns = [
        'classification', 'filename', 'aspect_ratio', 'aspect_ratio_elip', 'extreme_points',
        'contour_area', 'contour_perimeter', 'filled_circular_area_ratio',
        'complexity', 'circularity', 'roundness', 'perim_area_ratio',
        'solidity', 'equiv_d', 'laplacian'
    ]
    df_labeled = pd.DataFrame(record_list, columns=columns)
    output_dir = '/home/jko/ssl-cpi-analysis/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'labeled_data_geo_features.parquet')
    df_labeled.to_parquet(output_path)
    print(f"Saved labeled data to {output_path}")

if __name__ == "__main__":
    main()