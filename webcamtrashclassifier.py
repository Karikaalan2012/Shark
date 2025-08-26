# These are like 'ingredients' for our program. We are bringing in tools we need.
# 'tensorflow' is the main brain for our trash-sorting robot.
# 'cv2' (OpenCV) is for the robot's 'eyes' and for showing images.
# 'numpy' helps with math and working with images as numbers.
# 'os' helps us talk to the computer's folders and files.
# 'time' is for pausing the robot so it doesn't work too fast.
# 'datetime' is for putting a timestamp on our results.
import tensorflow as tf
import cv2
import numpy as np
import os
import time
from datetime import datetime

# --- Your Existing Class Definitions and Functions ---
# These are lists of all the different types of trash our robot can recognize.
# The number at the end tells us how many classes are in each list.
TRASH_CLASSES_6 = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
TRASH_CLASSES_5 = ['cardboard', 'metal', 'paper', 'plastic', 'trash']
TRASH_CLASSES_BINARY = ['recyclable', 'non-recyclable']


# This function checks how many types of trash the 'brain' (model) can handle.
# It looks at the model's 'output' to see what it's ready to tell us.
def detect_model_classes(interpreter):
    """Detects how many classes the model actually outputs"""
    output_details = interpreter.get_output_details()
    output_shape = output_details[0]['shape']

    if len(output_shape) == 2:
        return output_shape[1]
    else:
        return 1


# This function gets an image ready for the 'brain' to understand.
# It's like putting on a pair of glasses so the robot can see clearly.
def preprocess_image(img, input_details):
    """Preprocess image according to model requirements"""
    # The brain likes a certain color style, so we change the image's colors to match.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    # The brain needs a specific size, so we shrink or stretch the image to fit.
    img_resized = cv2.resize(img_rgb, (width, height))
    input_dtype = input_details[0]['dtype']

    # We change the numbers that make up the image to the type the brain likes.
    if input_dtype == np.uint8:
        img_preprocessed = img_resized.astype(np.uint8)
    elif input_dtype == np.int8:
        img_preprocessed = (img_resized.astype(np.float32) - 128).astype(np.int8)
    else:
        # We make the numbers a percentage (from 0 to 1) for the brain.
        img_preprocessed = img_resized.astype('float32') / 255.0

    # We give the image an extra dimension that the brain expects.
    return np.expand_dims(img_preprocessed, axis=0)


# This is the main function that does the trash sorting!
def predict_trash_image(model_path, image_path, class_names=None, show_image=False, save_results=False):
    """
    Predict trash classification using TensorFlow Lite model
    """
    # We first check if the 'brain' file and the image file actually exist.
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # We use a try/except block to catch any errors and not crash the program.
    try:
        # We load the 'brain' into our robot.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        num_classes = detect_model_classes(interpreter)

        # We decide which list of names to use based on the 'brain' we loaded.
        if class_names is None:
            if num_classes == 6:
                class_names = TRASH_CLASSES_6
            elif num_classes == 5:
                class_names = TRASH_CLASSES_5
            elif num_classes == 2:
                class_names = TRASH_CLASSES_BINARY
            else:
                class_names = [f'Class_{i}' for i in range(num_classes)]

        # The robot 'sees' the image.
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image")
            return None

        # We get the image ready for the 'brain'.
        img_batch = preprocess_image(img, input_details)

        # We give the image to the brain and ask it to think about it.
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()

        # We get the brain's answer!
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if len(output_data.shape) == 2 and output_data.shape[1] > 1:
            # We find the brain's top guess (the one with the highest confidence).
            predicted_class_idx = np.argmax(output_data[0])
            confidence = np.max(output_data[0])
            if predicted_class_idx < len(class_names):
                predicted_class_name = class_names[predicted_class_idx]
            else:
                predicted_class_name = f"Class_{predicted_class_idx}"

            # We print out a cool report with all the details.
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results_summary = f"""
=== TRASH CLASSIFICATION RESULTS ===
Time: {timestamp}
Image: {os.path.basename(image_path)}
Prediction: {predicted_class_name.upper()}
Confidence: {confidence:.4f} ({confidence * 100:.2f}%)
All predictions:"""
            for i, score in enumerate(output_data[0]):
                name = class_names[i] if i < len(class_names) else f"Class_{i}"
                results_summary += f"\n  {name:<15}: {score:.4f} ({score * 100:.2f}%)"
            results_summary += f"\n\n♻️  DISPOSAL RECOMMENDATION:"

            # Based on the guess, we give a recommendation on how to dispose of the trash.
            if num_classes == 2:
                if predicted_class_idx == 0:
                    results_summary += "\n    ✅ RECYCLABLE - Place in recycling bin"
                else:
                    results_summary += "\n    🗑️  NON-RECYCLABLE - Dispose in regular waste bin"
            else:
                recyclable_materials = ['cardboard', 'paper', 'plastic', 'metal', 'glass']
                if predicted_class_name.lower() in recyclable_materials:
                    results_summary += f"\n    ✅ This appears to be {predicted_class_name} - likely RECYCLABLE"
                elif predicted_class_name.lower() in ['organic', 'compost']:
                    results_summary += f"\n    🌱 This appears to be organic waste - COMPOSTABLE"
                else:
                    results_summary += f"\n    🗑️  This appears to be general trash - dispose in regular waste bin"
            results_summary += "\n" + "=" * 50
            print(results_summary)

            # We save the results to a file for later.
            if save_results:
                log_filename = "trash_classification_log.txt"
                with open(log_filename, 'a', encoding='utf-8') as log_file:
                    log_file.write(results_summary + "\n\n")

            # We can also show the image in a pop-up window.
            if show_image:
                cv2.imshow('Trash Classification Result', img)
                print("\nPress any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return predicted_class_name, confidence
        else:
            print("Unexpected output format")
            return None
    except Exception as e:
        # If there's an error, we print it out so we can figure out what went wrong.
        print(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# This function is a simple test to make sure our webcam 'eyes' are working.
def test_webcam():
    """Test function to check if webcam is working properly"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    ret, frame = cap.read()
    if ret:
        print("Webcam test successful!")
        cv2.imwrite('test_image.jpg', frame)
        print("Test image saved as 'test_image.jpg'")
    else:
        print("Error: Could not capture test frame")
        return False
    cap.release()
    return True


# This function captures a single picture from the webcam when you're ready.
def capture_image_from_webcam(filename="webcam_capture.jpg"):
    """Capture single image from webcam for manual classification"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return None
    print("Press 'Space' to capture image, 'Esc' to cancel")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        # We show you what the webcam sees in a window.
        cv2.imshow('Webcam - Press Space to Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # The 'Esc' key
            print("Capture cancelled")
            break
        elif key == 32:  # The 'Space' bar
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            break
    cap.release()
    cv2.destroyAllWindows()
    return filename if os.path.exists(filename) else None


# This function runs the robot automatically every few minutes.
def capture_and_classify_continuously(model_path, interval_minutes=5):
    """
    Captures images from webcam every specified interval and classifies them automatically.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    # We set the video's size to be high-quality.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # We create a folder to save all the pictures.
    if not os.path.exists('webcam_captures'):
        os.makedirs('webcam_captures')

    print(f"Starting automated trash classification...")
    print(f"Capturing and classifying images every {interval_minutes} minutes")
    print("Press Ctrl+C to stop")

    try:
        # This is the main loop that runs forever until you stop it.
        while True:
            ret, frame = cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_captures/capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\nImage captured: {filename}")
                result = predict_trash_image(model_path, filename, save_results=True)
                if result:
                    class_name, confidence = result
                    print(f"Classification complete: {class_name} ({confidence * 100:.2f}% confidence)")
                else:
                    print("Classification failed")
                print(f"Waiting {interval_minutes} minutes for next capture...")
                # We tell the robot to 'sleep' for the right amount of time.
                time.sleep(interval_minutes * 60)
            else:
                print("Error: Failed to capture frame")
                break
    except KeyboardInterrupt:
        # This part runs if you press Ctrl+C to stop the program.
        print("\nAutomatic capture and classification stopped by user")
    finally:
        # We always clean up the webcam to make sure nothing is still running.
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released")


# This is the 'main control center' of the program. It decides what to do first.
if __name__ == "__main__":
    # We tell the program where the 'brain' file is located.
    MODEL_PATH = "trash_classifier_model.tflite"
    print("Automated Trash Classification System")
    print("=" * 50)

    # We check if the 'brain' file is there before we do anything else.
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable with the correct path to your .tflite model")
        exit(1)

    # We show a menu and ask you what you want to do.
    print("\nChoose operation mode:")
    print("1. Single image classification (existing file)")
    print("2. Single capture and classify (webcam)")
    print("3. Continuous automated capture and classification")
    print("4. Test webcam")

    choice = input("Enter choice (1-4): ").strip()

    # We check your choice and then run the right function.
    if choice == "1":
        image_path = input("Enter image file path: ").strip()
        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}")
            exit(1)
        result = predict_trash_image(MODEL_PATH, image_path, show_image=True)
    elif choice == "2":
        image_path = capture_image_from_webcam()
        if image_path:
            result = predict_trash_image(MODEL_PATH, image_path, show_image=True)
        else:
            exit(1)
    elif choice == "3":
        if not test_webcam():
            print("Please check your webcam connection and try again")
            exit(1)
        try:
            interval = int(input("Enter capture interval in minutes (default 5): ") or "5")
            if interval <= 0:
                interval = 5
        except ValueError:
            interval = 5
        print(f"Starting continuous mode with {interval}-minute intervals...")
        capture_and_classify_continuously(MODEL_PATH, interval)
    elif choice == "4":
        if test_webcam():
            print("Webcam is working properly!")
        else:
            print("Webcam test failed. Please check your camera connection.")
    else:
        print("Invalid choice")
        exit(1)
