from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)



# Function to reduce the number of colors in the image using the K-means clustering algorithm
def change_color_quantization(img, k):
    
    # Read the image and transform it into a NumPy 3D array representing the B (blue), G (green), and R (red) values.
    data = np.float32(img).reshape((-1, 3))

    # Initialize the k-means clustering algorithm and defines its termination criteria. The maximum number of iterations is set to 20.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Apply the k-means clustering algorithm. The parameter k specifies the number of distinct colors in the output image.
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert centroid to unsigned 8-bit integer.
    center = np.uint8(center)

    # apply centroid values to all pixels by replacing their values with the center value.
    result = center[label.flatten()]

    # reshape the processed result into the shape of the original image.
    result = result.reshape(img.shape)

    # return resultant image
    return result

# Function to add cartoon affect to input image
def cartoonify_img(input_img,value):
   
    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Apply some median blur to smooth the image
    smooth_gray_frame = cv2.medianBlur(src=gray_frame, ksize=5)

    # Apply adaptive thresholding to get edges
    smooth_gray_frame_edges = cv2.adaptiveThreshold(src=smooth_gray_frame
                                                    , maxValue=255
                                                    # ADAPTIVE_THRESH_MEAN_C -> Mean of the neighborhood area
                                                    , adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C
                                                    # Type of threshold applied
                                                    , thresholdType=cv2.THRESH_BINARY
                                                    # Size of the neighbourhood area
                                                    , blockSize=9
                                                    # A constant subtracted from the mean of the neighborhood pixels
                                                    , C=9)
    

    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion
    smooth_gray_frame_edges = cv2.erode(smooth_gray_frame_edges, kernel, iterations=1)


    if value == 1:
        return smooth_gray_frame_edges
    
    # Enhance the blur and reduce the sharpness effects of the image.
    filtered_frame = cv2.bilateralFilter(src=input_img
                                         , d=9
                                         # Standard deviation of the filter in color space.
                                         , sigmaColor=200
                                         # Standard deviation of the filter in coordinate space.
                                         , sigmaSpace=200
                                         )
    

    # Combine the edge mask  with the filtered frame
    cartoon_frame = cv2.bitwise_and(filtered_frame, filtered_frame, mask=smooth_gray_frame_edges)

    # Applying color quantization to reduce the number of colors
    cartoon_frame = change_color_quantization(cartoon_frame, value)
    
    return cartoon_frame












image_urls = ['image1.png', 'image2.png', 'image3.png']

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the uploaded file from the request
        uploaded_file = request.files['image']

        # Save the file to a folder (you may need to create the folder)
        uploaded_file.save(uploaded_file.filename)

        image = cv2.imread(uploaded_file.filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = cartoonify_img(image,1)
        image2 = cartoonify_img(image,2)
        image3 = cartoonify_img(image,20)

        cv2.imwrite('static/image1.png', image1)
        cv2.imwrite('static/image2.png', image2)
        cv2.imwrite('static/image3.png', image3)

        # Send a JSON response back to the client
        return jsonify({'message': 'File uploaded successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/success')
def success():
   # image_urls = url_for('static', filename='satic/0001.png')
    return render_template('success.html', image_urls=image_urls)

if __name__ == '__main__':
    app.run(debug=True)

