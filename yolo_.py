import os
import time

# Start time
start_time = time.time()

from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # load a pretained model

# Use the model
results = model.train(data='split_data/', epochs=5)  # train the model
end_time = time.time()

# Calculate consumption time
consumption_time = end_time - start_time

print(f"Time taken to execute the code: {consumption_time} seconds")