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
The frames are stored at the following folder:
	- {project_folder}/frames/

Then we could perform feature matching and homography computation separately for both the left and right frames. We used several pruning techniques in addition to the results returned by SIFT. Several pairs of frames have been tested and the best results are written to the disk as homography.left and homography.right.

Once the program is launched, the image pair will be stacked side by side and different functions could be called by pressing the following keys:
	- q	quit
	- f	find matched points using SIFT
	- c	clear matches
	- s	show stitched image pair

Please note our program cannot find both homographies at one time; it has to process two images, for example, left and middle, before proceeding to find the homography of middle and right image. In default setting, it finds homography.right. If one would like to find homography.left, the program has to be modified in the following way:
	1. comment out line 249, 250 and 276
	2. uncomment line 245, 246 and 275

----------------------
stitch.py
----------------------
This simple program adjusts colours of the three images to make them uniform and stitches all three frames together, using the homographies calculated from the previous step.

-----------------------
main.py
-----------------------
The main program that execute tracking. In the script, the tracking algorithm we are using is Mean Shift Algorithm.

This is how the program works:
    1. On the first frame of video, find out all the bounding boxes of each players, and classify those bounding boxes according to the players' color. Therefore, we have red_players, blue_players and yellow_players
    2. for all the rest frames of video, run:
        2.1 Execute the Mean Shift Algorithm to get the new track window for each player
            2.1.1 For the original frame, extract the images within the court using court_mask
            2.1.2 After extract the court images, do color selection mask to select color such as red, blue ow yellow, according to the player's color.
            2.1.3 Do mean shift algorithm to get the new track window
        2.2 Adjust Tracking Windows to avoid the overlapping of the same color players' track windows
            2.2.1 For all tracking windows, select players whose tracking windows are overlaping with each other.
            2.2.2 For each pair of players with overlapped tracking windows, see which players' motion has more dramastic change such as distance change and angle change.
            2.2.3 For the player with more dramastic change, ignore the track window obtained by Mean Shift Algorithm, and create a new tracking window by predicting the possible positions of the player based on the motion history
            2.2.4 For the player with less dramastic change, continue using the track window obtained by Mean Shift Algorithm
        2.3 Draw the bounding boxes with color for each player
        2.4 Do homography mapping for each player on the top down view
Please run the script in this way:
    python main.py

Make sure that you have
    1. "output_frames" folder for that is the folder holding the stitched image outputs.
    2. "court_mask.jpg" for that filter out all the unrelated parts of the video