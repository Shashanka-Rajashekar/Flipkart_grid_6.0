from __future__ import division
import time
import cv2
import torch  # PyTorch for loading YOLOv5 model
import numpy as np
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv
from markdown2 import markdown  # Use markdown2 for Markdown rendering in VSCode
from PIL import Image  # Import PIL to work with images
import io  # For handling byte streams

# Initialize YOLOv5 model (downloads weights if not available locally)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

load_dotenv()

# Get the Google API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Check if the API key is available
if GOOGLE_API_KEY is None:
    raise ValueError("Google API Key not found. Set it in your environment variables.")

# Configure the Generative AI API
genai.configure(api_key=GOOGLE_API_KEY)

# Function to capture an image from mobile phone IP webcam
def capture_image_from_ipwebcam(ip_address):
    camera_url = f"http://{ip_address}/shot.jpg"  # Get a snapshot URL
    img_resp = requests.get(camera_url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    image = cv2.imdecode(img_arr, -1)
    
    if image is not None:
        return image
    else:
        print("Failed to capture image.")
        return None

# Function to detect products using YOLOv5
def get_product_details(image, yolo_model):
    # Save the captured image temporarily to pass to YOLOv5
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image)
    
    results = yolo_model(temp_image_path)
    detections = results.pandas().xyxy[0]
    
    products = {}
    for _, row in detections.iterrows():
        product_name = row['name']
        products[product_name] = products.get(product_name, 0) + 1
    
    return products

# Function to analyze image using Google Generative AI
def analyze_image_with_generative_ai(image_path):
    # Open the image file
    with open(image_path, "rb") as img_file:
        img = img_file.read()

    try:
        # Convert the bytes to a PIL image
        pil_image = Image.open(io.BytesIO(img))

        # Prompt the model with text and image content for analysis
        response = genai.GenerativeModel(model_name="gemini-1.5-pro-latest").generate_content(
            [pil_image, "Just give me the brand name ."]
        )

        # Extract the brand name from the response
        brand_name = None
        
        # Check if the response contains candidates
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            brand_name = response.candidates[0].content.parts[0].text.strip()

        if brand_name:
            return brand_name
        else:
            raise ValueError("Brand name not found in the response.")
    
    except Exception as e:
        print(f"Error analyzing image with Google Generative AI: {e}")
        return None

# Capture the image from IP Webcam continuously
ip_address = '192.168.29.101:8080'  # Replace with your IP Webcam address

# Dictionary to keep track of detected products and their counts
product_count = {}

while True:
    image = capture_image_from_ipwebcam(ip_address)

    if image is not None:
        # Use YOLOv5 to detect products
        detected_products = get_product_details(image, yolo_model)
        
        print("\nDetected Products using YOLOv5:")
        for product, count in detected_products.items():
            if product in product_count:
                product_count[product] += count  # Increment existing count
            else:
                product_count[product] = count  # Set new count
            print(f"{product_count[product]} {product}(s) detected.")

        # Save the captured image for analysis
        image_path = "temp_analysis_image.jpg"
        cv2.imwrite(image_path, image)

        # Analyze the image using Google Generative AI for product information extraction
        analysis_response = analyze_image_with_generative_ai(image_path)

        # Display the extracted brand name
        if analysis_response:
            print(f"Extracted Brand Name: {analysis_response}")
        else:
            print("No response received from the analysis.")

        # Optionally, display the image
        cv2.imshow("Captured Image", image)

    # Check for Esc key to exit
    if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
        break

cv2.destroyAllWindows()
