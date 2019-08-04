import cv2
import numpy as np

def get_contrasted(image, type="dark", level=3):
    maxIntensity = 255.0 # depends on dtype of image data
    phi = 1
    theta = 1

    if type == "light":
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
        newImage0 = np.array(newImage0,dtype='uint8')
        return newImage0
    elif type == "dark":
        newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**level
        newImage1 = np.array(newImage1,dtype='uint8')

        return newImage1

def sharp(image, level=3):
    f = cv2.GaussianBlur(image, (level,level), level)
    f = cv2.addWeighted(image, 1.5, f, -0.5, 0)
    return f

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
while 1:
    cropped_image = []
    ret,img = cap.read()
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
        
    if img is []:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces != ():
        for (x,y,w,h) in faces: 
            cv2.rectangle(img,((x-w+25),(y-70)),((x+(2*w)-25),(y+(2*h)+70)),(255,255,0),2)
            cropped_image = img[(y-70):(y+(2*h)+70),(x-w+25):(x+(2*w)-25)]
            gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            gray_img = sharp(get_contrasted(gray_img))
            gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            gray_blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
            adapt_thresh_im = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
            max_thresh, thresh_im = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            thresh = cv2.bitwise_or(adapt_thresh_im, thresh_im)


            gray = cv2.Canny(thresh, 88, 400, apertureSize=3)
            gray = cv2.dilate(gray, None, iterations=8)
            gray = cv2.erode(gray, None, iterations=8)

            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_info = []
            for c in contours:
                contour_info.append((
                    c,
                    cv2.isContourConvex(c),
                    cv2.contourArea(c),
                ))
            contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
            max_contour = contour_info[0]
            holes = np.zeros(gray_img.shape, np.uint8)
            cv2.drawContours(holes, max_contour, 0, 255, -1)

            mask = cv2.GaussianBlur(holes, (15, 15), 0)
            mask = np.dstack([mask] * 3)  # Create 3-channel alpha mask

            mask = mask.astype('float32') / 255.0  # Use float matrices,
            img1 = cropped_image.astype('float32') / 255.0  # for easy blending
            masked = (mask * img1) + ((1 - mask) * (0,0,1))  # Blend
            masked = (masked * 255).astype('uint8')

        cv2.imshow('crpimg',gray)
        # Wait for Esc key to stop 
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    else:
        continue

# Close the window 
cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 
