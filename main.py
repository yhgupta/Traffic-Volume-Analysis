import cv2
import numpy as np
import vehicle
from vehicle import Vehicle

# All the required files location
FILES_PATH = './model'

# These are the required files to run the YOLO model
YOLO_CFG_PATH = FILES_PATH + '/yolov3.cfg'
YOLO_TXT_PATH = FILES_PATH + '/yolov3.txt'
YOLO_WEIGHTS_PATH = FILES_PATH + '/yolov3.weights'

# Path to sample video, make this 0 (int) for webcam stream
SAMPLE_VIDEO_PATH = './samples/output.mkv'

# Set up prediction classes
with open(YOLO_TXT_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# These are the classes we require
req_classes = [vehicle.BICYCLE, vehicle.CAR, vehicle.MOTORCYCLE, vehicle.BUS, vehicle.TRUCK]

# Initialize the neural network with the model configurations and weights
net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)

# Gets the output of the neural network
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Color of the rectangles on the vehicles
COLOR_DIR_DOWN = (50,205,50)
COLOR_DIR_UP = (220,20,60)
COLOR_DIR_NONE = (0,0,128)

# Array of colors
colors = [COLOR_DIR_DOWN, COLOR_DIR_UP, COLOR_DIR_NONE]

# Draws specific color rectangle on the vehicle
def draw_prediction(img, class_id, direction, x, y, x_plus_w, y_plus_h):

    # Gets the label of class id, E.g. Bicycle has class id 1
    label = str(classes[class_id])

    # See vehicle.py for values of direction
    color = colors[direction]

    # Draws a rectangle over the vehicle
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    # Draws a label stating the type of vehicle above the box
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# A vehicle count map of for {'VEHICLE_TYPE': [DOWN_COUNT, UP_COUNT, NONE_COUNT]}
count = {vehicle.V_LIGHT_VEHICLE: [0, 0, 0], vehicle.LIGHT_VEHICLE: [0, 0, 0], vehicle.HEAVY_VEHICLE: [0, 0, 0]}

# Draws the count of vehicles on the top left of screen
def draw_count(img):
    cv2.putText(img, 'UP', (16,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)
    cv2.putText(img, 'V Light: ' + str(count[vehicle.V_LIGHT_VEHICLE][vehicle.DIR_UP]), (16,41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)
    cv2.putText(img, 'Light  : ' + str(count[vehicle.LIGHT_VEHICLE][vehicle.DIR_UP]), (16,66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)
    cv2.putText(img, 'Heavy  : ' + str(count[vehicle.HEAVY_VEHICLE][vehicle.DIR_UP]), (16,91), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)

    cv2.putText(img, 'DOWN', (200,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)
    cv2.putText(img, 'V Light: ' + str(count[vehicle.V_LIGHT_VEHICLE][vehicle.DIR_DOWN]), (200,41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)
    cv2.putText(img, 'Light  : ' + str(count[vehicle.LIGHT_VEHICLE][vehicle.DIR_DOWN]), (200,66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)
    cv2.putText(img, 'Heavy  : ' + str(count[vehicle.HEAVY_VEHICLE][vehicle.DIR_DOWN]), (200,91), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)

# Array of vehicles kept in memory
vehicles = []

# Checks a given bounding box and vehicle
# Removes a vehicle from vehicles array based on frame expiration
# Match and update a vehicle based on bounding box
# Create a new vehicle if none
def check_vehicle(C_FRAME, v_type, x, y, w, h):
    # Variable to check whether the vehicle is already present in array
    present = False
    for vehicle in vehicles:
        if vehicle.expired(C_FRAME):
            # If vehicle expired, increase the count and remove it
            t = vehicle.get_type()
            d = vehicle.get_direction()
            count[t][d] += 1
            vehicles.remove(vehicle)
            v = vehicle
            present = True
            break
        elif vehicle.match_and_update(C_FRAME, v_type, x, y, w, h):
            # If vehicle updated, its present
            v = vehicle
            present = True
            break
    # If none of vehicle matches the bounding box, create a new vehicle
    if not present:
        v = Vehicle(C_FRAME, v_type, x, y, w, h)
        vehicles.append(v)

    # returns the direction of vehicle used to draw the color of box
    return v.get_direction()

# Loads the video
cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)

# Current frame
C_FRAME = 0

# Read a single frame to get width and height info, if we want to write output video
ret, frame = cap.read()

# Initialize writers
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output/video4.avi',fourcc, 20.0, (frame.shape[1],frame.shape[0]))

# Run while we have a frame
while ret:
    C_FRAME += 1
    width = frame.shape[1]
    height = frame.shape[0]

    # Scale down a image, neural network does'nt need complete image
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    # Get the output of neural network
    outs = net.forward(get_output_layers(net))

    # Required variables
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Loop through output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            # Get class id with max score
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # If the class is one of required classes and have a 50% surity
            if class_id in req_classes and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply NMS to simplify overlapping boxes
    # See https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Loop through all bounding boxes, check the vehicle and draw the boxes and count
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        dir = check_vehicle(C_FRAME, class_ids[i], round(x), round(y), round(w), round(h))
        draw_count(frame)
        draw_prediction(frame, class_ids[i], dir, round(x), round(y), round(x+w), round(y+h))

    # Uncomment the output line if you want to save the video as well
    #output.write(frame)
    cv2.imshow("Traffic Volume Analysis", frame)

    ret, frame = cap.read()
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all the variables
cap.release()
output.release()
cv2.destroyAllWindows()