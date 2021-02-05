# People Counter
-Will Follow-

## How it works
-Will Follow-

The underlying algorithm works in the following manner. A object detection algorithm is run every n-th frame. In-between the detections, a correlation tracker will track the objects until the next round of detection. The results of the detection step will be associated with the tracked objects by simply computing Euclidean distances and assigning the closest objects to each other. Face detection is run *once* after m frames of the first detection.

## Usage
-Will Follow-

*Explanation of the configurable parameters*

Parameter | Description
--------- | -----------
Confidence | Persons are only considered if the detection certainty is above this threshold.
Skip Frames | Amount of Frames to skip between running the detection algorithm. In-between detections, a correlation tracking algorithm takes care of the tracking. Lower values improve accuracy, but requires more computational power.
Dimension | The dimension to resize the video's larger dimension to prior to running the script.
Count Direction | Either "vertical" or "horizontal". Whether to count people in the vertical or horizontal direction.
Counting Line Position | Specifies at what position people are counted. Must be between 0 and 1. Values close to 0 position the line near the top/left, while values close to 1 position the line near the bottom, right. A fraction of the video's height/width. Can also make the algorithm more robust when positioning the line closer to the camera where detections are more accurate.
Minimum #Frames before Count | The minimum amount of frames for which a person must have been tracked, before it is counted when crossing the line. 
Draw Bounding Boxes | Either "true" or "false". If "true", bounding boxes are drawn onto the frame, with no additional information. If "false", the persons centroids are drawn, together with their id, gender and age, if applicable.
Max Disappeared | Maximum number of frames to keep looking for a person to re-appear. When he/she has left the frame or is lost.
Max Distance | The maximum distancce to re-assign the same id to a person. When the tracking and subsequent detection is off by more than this value, a new id will be assigned to the person.
Analyze Face | Either "true" or "false". Whether to run facial attributes analysis. If "false", all ensuing parameters are ignored.
Enforce Detection | Either "true" or "false". If "true", the facial analysis will only be run, if a face is detected inside the person's bounding box. If "false", the analysis will be run, even if no face was detected.
Detection BB Padding | Amount of pixels to pad the bounding boxes with prior to inputting it into the deepface model.
Analyze After | Amount of frames to wait, before running the facial analysis algorithm on a detected person.

## Setup
- Will Follow -
