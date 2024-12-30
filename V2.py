import cv2
from ultralytics import YOLO
import torch

#model = YOLO('C:/Users/mohse/Desktop/Project/Test/Test/runs/detect/face_anti_spoofing/weights/best.pt')  # Replace 'best.pt' with the path to your trained YOLO model
#model.export(format='tflite',imgsz=640, save_dir=r"C:/Users/mohse/Desktop/Project/Test/Test/Trained_Models")  # TensorFlow SavedModel



def train_yolo():
    # Paths to the dataset and configuration file
    dataset_path = r"C:/Users/mohse/Desktop/Project/Test/Test/Dataset_Altered/data.yaml"  # Your .yml file should be named `data.yaml` and located in the working directory.

    # Initialize YOLO model
    model = YOLO('yolo11n.pt')  # Replace 'yolov8n.pt' with the desired pre-trained model version (n/s/m/l/x).

    # Train the model using GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train(
        data=dataset_path,  # Path to dataset configuration
        epochs=50,  # Train for sufficient epochs
        imgsz=640,  # Standard input size for YOLO
        batch=8,  # Reduce batch size for RTX 3050 Ti if necessary
        project='YOLO11_Training',  # Project name for logging
        name='experiment_name',  # Experiment name
        augment=True,  # Enable data augmentation
    )
 
    print("Training completed. Check the runs folder for results.")
train_yolo()

def live_prediction2():
    import cv2
    from ultralytics import YOLO

    # Load the YOLO model
    model = YOLO('C:/Users/mohse/Desktop/Project/Test/Test/YOLO11_Training/experiment_name3/weights/best.pt')

    # Initialize video capture for live prediction
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    # Set video capture properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform prediction using YOLO
        results = model.predict(source=frame, conf=0.6, iou=0.4, show=False)  # Adjust confidence and IoU thresholds

        # Display the YOLO layout
        results_img = results[0].plot()  # Use YOLO's built-in plotting method

        # Show the annotated frame
        cv2.imshow('Live Face Anti-Spoofing Detection', results_img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

live_prediction2()









