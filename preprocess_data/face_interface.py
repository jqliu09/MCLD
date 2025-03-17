from insightface.app import FaceAnalysis
import os 
import cv2
import numpy as np

def expand_image_region(box, scale_factor=1.5):
    x1, y1, x2, y2 = box
    h, w = (y2 - y1)*scale_factor, (x2 - x1)*scale_factor
    center_x, center_y = (x1 + x2) / 2 , (y1 + y2) / 2
    length_ = h if h > w else w
    x1, x2 = center_x - length_ / 2, center_x + length_ / 2
    y1, y2 = center_y - length_ / 2, center_y + length_ / 2 
    return np.array([x1, y1, x2, y2], dtype=int)

def safe_crop_image(image, box):
    (x1, y1, x2, y2) = box.astype("int")
    h, w = image.shape[:2]
    if x1 > 0 and y1 > 0 and x2 < w and y2 < h:
        return image[y1:y2, x1:x2, :]
    else:
        padded_image = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        start_x = max(0, -x1)
        start_y = max(0, -y1)
        
        # end_x = min(w - x1, x2 - x1)
        # end_y = min(h - y1, y2 - y1)
        cropped_region = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        padded_image[start_y:start_y + cropped_region.shape[0], start_x:start_x + cropped_region.shape[1]] = cropped_region
        return padded_image    

if __name__ == "__main__":
    
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    image_path = './dataset/fashion/train/'
    image_names = os.listdir(image_path)
    
    super_resolution = True
    
    for image_name in image_names:
        image = cv2.imread(image_path + image_name)
        face_info = app.get(image)
        if len(face_info) == 0:
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        if not super_resolution:
            face_emb = face_info['embedding']
        else:
            face_bbox = face_info['bbox'].astype(int)
            bbox = expand_image_region(face_bbox) 
            # Crop the face region and resize to clip size 224 * 224
            face = safe_crop_image(image, bbox)
            out = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        print("aaaa")