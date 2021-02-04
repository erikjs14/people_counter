# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import json
from time import time
from math import floor
from os.path import dirname, join
import os
from sys import maxsize, platform
from google.cloud import storage

"""## Helper"""

# centroid tracking algorithm implementation
# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True)
ap.add_argument('-g', '--credentials-path', type=str)
ap.add_argument('-o', '--output', type=str)
ap.add_argument('-p', '--progress-update', type=int, default=3000)
ap.add_argument('-c', '--confidence', type=float, default=0.4)
ap.add_argument('-s', '--skip-frames', type=int, default=30)
ap.add_argument('-d', '--dimension', type=int, default=500) # dimension to which to crop the video to
ap.add_argument('-a', '--count-direction', type=str, default='vertical') # whether to count in horizontal or vertical direction
cargs = vars(ap.parse_args())

# Arguments, can be implemeted as command line args outside this notebook
args = {
  'prototxt': 'MobileNetSSD_deploy.prototxt', # path to Caffe deploy prototxt file
  'model': 'MobileNetSSD_deploy.caffemodel', # path to pre-trained Caffe model
#   'input': cargs['input'], # input video
#   'output': cargs['output'], # output video
  'object-to-track': 'person', # class name of the to-track object type
}
args = { **args, **cargs };

if platform == 'linux': os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args['credentials_path']

# https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-code-sample
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


# centroid tracking alg taken from: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects

#class representing an object to be tracked
class TrackableObject:
  def __init__(self, id, centroid):
    # unique id of this object
    self.id = id
    # list of centroids during tracking
    self.centroids = [centroid]
    # whether this object has already been counted
    self.counted = False

"""## Tracking Implementation

Perform object detection only once every _skip-frames_ frames, and run correlation filters for object tracking in between. Object detection is assigned to tracked objects by centroid tracking logic. The algorithm is in one of three states:
1. _Waiting_: Idle
2. _Detecting_: Performing object detection using the MobileNet
3. _Tracking_: Tracking objects and counting without further detection.

Object Detection Model: MobileNet SSD
"""

# MobileNet classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load model
protoPath = join(dirname(__file__), args['prototxt'])
modelPath = join(dirname(__file__), args['model'])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print('Loaded Model')

# load video
inputPath = join(dirname(__file__), args['input'].strip())
video = cv2.VideoCapture(inputPath)
amountFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT));
originalFps = int(video.get(cv2.CAP_PROP_FPS))
print('Loaded Video')

# init output tmp file path
tmp_out_path = join(dirname(__file__), 'tmp_output', args['output'].strip()) if args['output'] is not None else None
if not os.path.exists(join(dirname(__file__), 'tmp_output')):
    os.makedirs(join(dirname(__file__), 'tmp_output'))

# init video writer
writer = None

# frame dimensions
W = H = None

# init centroid tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = [] # stores dlib correlation trackers
trackableObjects = {} # maps object ids to a trackableObject

totalFrames = 0
totalUp = 0
totalDown = 0
totalLeft = 0
totalRight = 0
totalCounted = 0

fps = FPS().start()

now = lambda: floor(time() * 1000);

# loop over frames and perform algorithm
lastChecked = now()
while True:
  # get frame and stop if at end
  frame = video.read()[1]
  if frame is None: 
    break

  # reduce dimensions (for performance)
  # and convert from BGR to RGB
  frame = imutils.resize(frame, width=args['dimension'])
  #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # set frame dimensions (only once)
  if W is None or H is None:
    (H, W) = frame.shape[:2]

  # init writer
  if args['output'] is not None and writer is None:
    fourcc = cv2.VideoWriter_fourcc(*'VP90')#(*'AV10')#(*'MJPG')#(*'H264')
    writer = cv2.VideoWriter(tmp_out_path, fourcc, 30, (W, H), True)

  # init current status
  status = 'waiting'
  # init list of bounding box rects
  rects = []

  # run object detection if at n-th frame
  if totalFrames % args['skip_frames'] == 0:
    status = 'detecting'
    trackers = []

    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(W, H), mean=127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over detections and find relevant objects
    for i in np.arange(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > args['confidence']:
        idx = int(detections[0, 0, i, 1]) # index of the class label
        if CLASSES[idx] != args['object-to-track']: # only care for persons
          continue

        # compute bounding box coords
        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype('int')
        
        # start correlation tracker and add to list of trackers

        ### dlib ###
        #tracker = dlib.correlation_tracker()
        #rect = dlib.rectangle(startX, startY, endX, endY)
        #tracker.start_track(rgb, rect)
        ### dlib ###
        ### opencv ###
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (startX, startY, endX-startX, endY-startY))
        ### opencv ###

        trackers.append(tracker)

  # run trackers instead of detector
  else:
    for tracker in trackers:
      status = 'tracking'

      # update tracker and get new position

      ### dlib ###
      #tracker.update(rgb)
      #pos = tracker.get_position()
      #startX = int(pos.left())
      #startY = int(pos.top())
      #endX = int(pos.right())
      #endY = int(pos.bottom())
      ### dlib ###
      ### opencv ###
      (succ, box) = tracker.update(frame)
      startX, startY, w, h = [int(v) for v in box]
      endX = startX + w
      endY = startY + h
      ### opencv ###

      # add bounding box to history
      rects.append((startX, startY, endX, endY))

  # draw line at which counting happens
  if writer is not None: 
    if args['count_direction'].strip() == 'vertical':
      cv2.line(frame, (0, H//2), (W, H//2), (255, 166, 0), 2)
    else:
      cv2.line(frame, (W//2, 0), (W//2, H), (255, 166, 0), 2)
  
  # utilize centroid tracker to associate objects and rect centroids
  objects = ct.update(rects)
  
  # loop over tracked objects for actual counting
  for (objectID, centroid) in objects.items():
    
    # check whether a trackable object exists
    to = trackableObjects.get(objectID, None)

    # if not, create a new one
    if to is None:
      to = TrackableObject(objectID, centroid)
    # else, use it to determine direction
    else: 
      # direction computed by difference of mean of old centroids and current centroid (y-coordinates)
      # negative is up/left and positive is down/right
      x = [c[0] for c in to.centroids]
      y = [c[1] for c in to.centroids]
      direction_x = centroid[0] - np.mean(x)
      direction_y = centroid[1] - np.mean(y)
      to.centroids.append(centroid)

      # check whether already counted
      if not to.counted:
        if args['count_direction'].strip() == 'vertical':
          # count, if: direction is up AND centroid is above center line (and not counted before)
          if direction_y > 0 and centroid[1] > H//2:
            totalDown += 1
            to.counted = True
          # equivalent for other direction
          if direction_y < 0 and centroid[1] < H//2:
            totalUp += 1
            to.counted = True
        else:
          if direction_x > 0 and centroid[0] > W//2:
            totalRight += 1
            to.counted = True
          # equivalent for other direction
          if direction_x < 0 and centroid[0] < W//2:
            totalLeft += 1
            to.counted = True

    # store object
    trackableObjects[objectID] = to

    # drawing object info on screen (id and position)
    if writer is not None:
      text = f'ID {objectID}'
      cv2.putText(frame, text, (centroid[0] -10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

  if writer is not None:
    # drawing general info on screen
    info = None
    if args['count_direction'].strip() == 'vertical':
      info = [
        ('Counted', ct.nextObjectID),
        ('Up', totalUp),
        ('Down', totalDown),
        ('Status', status),
      ]
    else:
      info = [
        ('Counted', ct.nextObjectID),
        ('Left', totalLeft),
        ('Right', totalRight),
        ('Status', status)
      ]
    for (i, (k, v)) in enumerate(info):
      text = f'{k}: {v}'
      cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    writer.write(frame)

  #cv2.imwrite('frame_{}.jpg'.format(str(totalFrames).zfill(5)), rgb)

  # show output frame
  #clear_output()
  #cv2_imshow(frame)
  #key = cv2.waitKey(1) & 0xFF

  # quit loop if key `q` is pressed
  #if key == ord('q'):
   # break

# check every five frames if update to be sent
  if totalFrames % 5 == 0:
    nowt = now()
    if nowt - lastChecked > args['progress_update']:
        lastChecked = nowt
        print(f'[PROGRESS%] {floor(totalFrames/amountFrames*100)}')
        print(f'Frames processed: {totalFrames}', flush=True)
  totalFrames += 1
  fps.update()

# clean up
fps.stop()
print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
if fps.elapsed() > 0: print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

# get amount of frames each object was in frame
# by counting the number of centroids in history
time_in_frame = {}
for to in list(trackableObjects.values()):
    time_in_frame[to.id] = len(to.centroids)
# get min, max and average time in frame
min = maxsize
max = 0
avg = 0
for val in list(time_in_frame.values()):
    min = val if val < min else min
    max = val if val > max else max
    avg += val
avg /= len(time_in_frame)
# from amount frames to seconds
min /= originalFps
max /= originalFps
avg /= originalFps


# results as json
results = [
    { 'label': 'Elapsed Time', 'value': '{:.2f}s'.format(fps.elapsed()) },
    { 'label': 'Approx. FPS', 'value': '{:.2f}s'.format(fps.fps()) },
    { 'label': 'Counted Up', 'value': totalUp },
    { 'label': 'Counted Down', 'value': totalDown },
    { 'label': 'Total Counted', 'value': ct.nextObjectID },
    { 'label': 'Min Frame Time', 'value': '{:.2f}s'.format(min) },
    { 'label': 'Max Frame Time', 'value': '{:.2f}s'.format(max) },
    { 'label': 'Avg Frame Time', 'value': '{:.2f}s'.format(avg) },
];
print('[RESULTS] ' + json.dumps(results));

if writer is not None:
  writer.release()
  upload_blob('pca-tmp-processed-video-storage', tmp_out_path, f'processed/{args["output"].strip()}')
video.release()
cv2.destroyAllWindows()