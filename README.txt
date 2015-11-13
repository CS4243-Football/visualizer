***************************
** Soccer Tacking System **
***************************

This system takes in 3 input videos from 3 cameras filming the left, middle and right portion of a soccer field respectively during an actual match. The 3 video sources are then stitched together to give an overall view of the whole field. Tracking is also performed on players from both teams (excluding the goal keepers, sideline referees and the football itself). These players are then labeled with their team colours and projected onto a separate top-down soccer field for further offsite animation and running distance estimation.


Video source - from the team comprising Dennis, Larry, Yujian and Charles
             - taken by 3 DSLR cameras

----------------------
find_homo.py
----------------------
Before running this program, we have to convert each video source into individual frames first. This can be performed using the following function:
	— saveFrames(vidName, frameName, scale)

Then we could perform feature matching and homography computation separately for both the left and right frames. We used several pruning techniques in addition to the results returned by SIFT. Several pairs of frames have been tested and the best results are written to the disk as homography.left and homography.right.

Please note our program cannot find both homographies at one time; it has to process two images, for example, left and middle, before proceeding to find the homography of middle and right image. 

----------------------
stitch.py
----------------------
This simple program adjusts colours of the three images to make them uniform and stitches all three images together, using the homographies calculated from the previous step.

-----------------------
main.py
-----------------------
