# Background-subtraction-of-live-feed

The objective is to determine the person facing the camera and extract the
foreground from the video frames such that the background noise is eliminated.

1. The idea is to fetch the frame from the video one at a time and detect the face of the
instructor from each frame.
2. According to the face detected, a bounding box is determined around it which take the
human face and body upto the abdomen. This will help to eliminate the excess
background noise from the video frame and the focus will only be on the instructor.
3. Later, contouring will determine the exact outline of the instructor which will help to
eliminate the rest of the noise from the background.
4. The final stage will be to stream the noise free video over web which can be further
enhanced to support over various browsers in future.


Installation:

To execute this code you should have Python 3.6 or above.
You will also need numpy and OpenCV libraries.

Numpy: pip install numpy

OpenCV: pip install opencv-python

After this you can directly run Background_Subtraction.py.

This file will only show you till thresolding part, as it is incomplete.
