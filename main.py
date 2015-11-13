import cv2
import numpy as np
from cv2 import cv
from sets import Set

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, track_window):
        c, r, w, h = track_window
        # self.top_left = Point(c, r)
        # self.bottom_right = Point(c + w, r + h)

        self.left = c
        self.right = c + w
        self.top = r
        self.bottom = r + h

    def top(self): 
        return self.top


    def bottom(self):
        return self.bottom

    def left(self):
        return self.left

    def right(self):
        return self.right

    def isOverlap(self, other):
        if self.left > other.right or other.left > self.right:
            return False
        
        if self.top > other.bottom or other.top > self.bottom:
            return False

        return True

    def area(self):
        return (self.right - self.left) * (self.bottom - self.top)

    def overlap_area(self, other):
        isOverlap = self.isOverlap(other)

        if isOverlap == False:
            return 0
        else:
            left = max(self.left, other.left)
            right = min(self.right, other.right)
            bottom = max(self.bottom, other.bottom)
            top = min(self.top, other.top)

            intersection_area = (right - left) * (bottom - top)
            return intersection_area

class Player:
    def __init__(self, total_track_window_size, color, mask_lower_bound, mask_upper_bound):
        self.track_windows = []
        self.color = color
        self.total_track_window_size = total_track_window_size
        self.mask_lower_bound = mask_lower_bound
        self.mask_upper_bound = mask_upper_bound
        self.adjusted_belfore = False
        self.temp_new_track_window = False

    def add_track_window(self, track_window):
        self.track_windows.append(track_window)

        while len(self.track_windows) > self.total_track_window_size:
            self.track_windows.pop(0)

    def get_current_track_window(self):
        last_index = len(self.track_windows) - 1
        return self.track_windows[last_index]

    def get_track_windows(self):
        return self.track_windows


    def is_adjusted_before(self):
        return self.is_adjusted_before()

    def set_temp_new_track_window(self, temp_new_track_window):
        self.temp_new_track_window = temp_new_track_window

    def get_temp_new_track_window(self):
        return self.temp_new_track_window

    def reset():
        self.temp_new_track_window = false

homography_matrix = [
    [6.45987489857, 23.3079670706, -11843.6959535],
    [0.310938858094, 44.8291015983, -5873.0617813],
    [0.000538489309133, 0.0697341889234, 1.0]
]

court_mask = cv2.imread("court_mask.jpg")
top_down_background_img = cv2.imread("topdownField.jpg")

def main():
    print "Main Function. Let the fun begin"
    mean_shift()

    # for i in range(0, 5000):
    #     frame = read_frame(i)
    #
    #     frame = cv2.bitwise_and(frame, court_mask)
    #     cv2.imshow('frame',frame)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #     # else:
    #     #     cv2.imwrite("track_{}.jpg".format(i), frame)



def mean_shift():
    frame = read_frame(0)

    all_players,red_players, blue_players, yellow_players = setup_all_players()

    draw_all_players_current_tracking_window(frame, all_players)

    cv2.imshow('frame',frame)
    cv2.imwrite("track_{}.jpg".format(0), frame)

    top_down_view(all_players, 0)
    cv2.waitKey(0)
    for i in range(1, 5000):
        print "Next Frame Number is ", i

        frame = read_frame(i)
        update_all_players_current_tracking_window(frame, all_players)
        justify_all_players_track_windows(red_players)
        justify_all_players_track_windows(blue_players)
        justify_all_players_track_windows(yellow_players)

        for i in range(len(all_players)):
            player = all_players[i]
            temp_new_track_window = player.get_temp_new_track_window()
            player.add_track_window(temp_new_track_window)
            player.set_temp_new_track_window(False)

        draw_all_players_current_tracking_window(frame, all_players)

        cv2.imshow('frame',frame)
        top_down_view(all_players, i)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite("track_{}.jpg".format(i), frame)

def setup_all_players():
    # red_players, blue_players = [], []
    all_players = []
    red_players = []
    blue_players = []
    yellow_players = []
    red_track_windows, blue_track_windows, yellow_track_windows = setup_all_track_windows()
    total_track_window_size = 8

    for i in range(len(red_track_windows)):
        track_window = red_track_windows[i]
        player = Player(total_track_window_size, (0, 0, 255), (17, 15, 100), (50, 56, 200))
        player.add_track_window(track_window)
        all_players.append(player)
        red_players.append(player)

    for i in range(len(blue_track_windows)):
        track_window = blue_track_windows[i]
        player = Player(total_track_window_size, (255, 0, 0), (40, 28, 4), (220, 187, 50))
        player.add_track_window(track_window)
        all_players.append(player)
        blue_players.append(player)

    for i in range(len(yellow_track_windows)):
        track_window = yellow_track_windows[i]
        player = Player(total_track_window_size, (0, 255, 255), (30, 150, 150), (70, 255, 255))
        player.add_track_window(track_window)
        all_players.append(player)
        yellow_players.append(player)

    return all_players, red_players, blue_players, yellow_players


def top_down_view(all_players, index):
    top_down_background_img_copy = top_down_background_img.copy()
    #=========== modified from here ===============
    height,width,channel = top_down_background_img_copy.shape
    x_coord_red  = []
    x_coord_blue = []
    for i in range(len(all_players)):
        player = all_players[i]
        color = player.color
        track_window = player.get_current_track_window()
        centroid = get_centroid(track_window)
        mapped_point = get_homography_mapped_point(centroid, homography_matrix)
        if (color == (0,0,255)):
            x_coord_red.append(mapped_point[0])
        else:
            x_coord_blue.append(mapped_point[0])
        cv2.circle(top_down_background_img_copy, mapped_point, 5, color, -1)

    offsite_red_x = min(x_coord_red)
    offsite_blue_x = max(x_coord_blue)
    offset = 5
    cv2.line(top_down_background_img_copy, (offsite_red_x-offset,0), (offsite_red_x-offset,height), (0,0,255), 2)
    cv2.line(top_down_background_img_copy, (offsite_blue_x+offset,0), (offsite_blue_x+offset,height), (255,0,0), 2)
    #========== modified end here =================
    cv2.imshow("Top Down View", top_down_background_img_copy)
    cv2.imwrite("view_{}.jpg".format(index), top_down_background_img_copy)


def get_homography_mapped_point(point, homography_matrix):
    x, y = point

    mapped_point = np.dot(homography_matrix, [[x], [y], [1]])

    return (int(mapped_point[0][0] / mapped_point[2][0]), int(mapped_point[1][0] / mapped_point[2][0]))


def get_mapped_centroid(track_window):
    centroid = get_centroid(track_window)
    mapped_point = get_homography_mapped_point(centroid, homography_matrix)

    return mapped_point


def get_centroid(track_window):
    c, r, w, h = track_window
    xc = c + w/2
    yc = r + h

    return (xc, yc)


def mean_shift_tracking_window(fgmask, track_window, n):
    if n == 0:
        return track_window
    c, r, w, h = track_window

    frame_window = fgmask[r:r+h, c:c+w]

    M00 = 0.0
    M01 = 0.0
    M10 = 0.0


    for i in range(h):
        for j in range(w):
            if len(frame_window) > i \
                    and len(frame_window[i]) > j \
                    and frame_window[i, j] > 0:
                M00 += 1#frame_window[i, j]
                M01 += j#* frame_window[i, j]
                M10 += i# * frame_window[i, j]

    if M00 == 0:
        return track_window

    xc = int(M01 / M00)
    yc = int(M10 / M00)

    if w/2 == xc and h/2 == yc:
        return track_window
    else:
        delta_c = xc - w/2
        delta_r = yc - h/2

        track_window = (c + delta_c, r + delta_r, w, h)

        return mean_shift_tracking_window(fgmask, track_window, n - 1)



def display_all_current_track_windows(all_players):
    for i in range(len(all_players)):
        player = all_players[i]
        print player.get_current_track_window()

def draw_rect_with_track_window(frame, track_window, color):
    x,y,w,h = track_window

    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)


def draw_all_players_current_tracking_window(frame, all_players):
    for i in range(len(all_players)):
        player = all_players[i]
        draw_rect_with_track_window(frame, player.get_current_track_window(), player.color)
        cv2.putText(frame, "{}".format(i), get_centroid(player.get_current_track_window()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)



def update_all_players_current_tracking_window(frame, all_players):
    for i in range(len(all_players)):
        player = all_players[i]
        print "Player Number", i
        update_track_window(frame, player)

def justify_all_players_track_windows(players):
    threshold = 0.7
    overlapped_players = Set([])
    for i in range(len(players)):
        this_player = players[i]

        if this_player not in overlapped_players:

            for j in range(len(players)):

                if i != j and players[j] not in overlapped_players:
                    other_player = players[j]
                    
                    this_rect = Rectangle(this_player.get_temp_new_track_window())
                    other_rect = Rectangle(other_player.get_temp_new_track_window())

                    overlap_area = this_rect.overlap_area(other_rect)
                    this_area = this_rect.area()
                    other_area = other_rect.area()

                    if float(overlap_area) / this_area > threshold or float(overlap_area) / other_area > threshold:
                        overlapped_players.add(this_player)
                        overlapped_players.add(other_player)

    for player in overlapped_players:
        justify_player_track_window(player)

def justify_player_track_window(player):
    player_track_windows = player.get_track_windows()

    most_recent_old_window = player_track_windows[len(player_track_windows) - 1]
    second_most_recent_old_window = player_track_windows[len(player_track_windows) - 2]
    
    x1, y1, w1, h1 = second_most_recent_old_window
    x2, y2, w2, h2 =  most_recent_old_window
    new_track_window = move_track_window(most_recent_old_window, (x2 - x1, y2 - y1))

    player.set_temp_new_track_window(new_track_window)
    
def justified_track_window(player, new_track_window):
    player_track_windows = player.get_track_windows()

    total_gradients = np.array([0, 0])
    initial_track_window = player_track_windows[0]
    most_recent_old_window = player_track_windows[len(player_track_windows) - 1]
    second_most_recent_old_window = player_track_windows[len(player_track_windows) - 2]


    for i in range(1, len(player_track_windows)):
        old_track_window = player_track_windows[i]
        total_gradients += np.array(get_gradient(initial_track_window, old_track_window))

    average_gradient = total_gradients / len(player_track_windows)
    temp_new_gradient = np.array(get_gradient(initial_track_window, new_track_window))
    angle = angle_between(average_gradient, temp_new_gradient)
    #

    new_dist = get_distance(most_recent_old_window, new_track_window)
    old_dist = get_distance(second_most_recent_old_window, most_recent_old_window)
    print "angle", angle, "new_dist", new_dist, "old_dist", old_dist
    if angle > 0 and angle < 180 and abs(new_dist - old_dist) > 4 and player.is_adjusted_before == False:#angle >= 90 and angle != 180:
        print "Justify track window"
        x1, y1, w1, h1 = player_track_windows[len(player_track_windows) - 2]
        x2, y2, w2, h2 =  most_recent_old_window
        new_track_window = move_track_window(most_recent_old_window, (x2 - x1, y2 - y1))
        player.is_adjusted_before = True

    return new_track_window



    # old_velocity = get_gradient_length(old_graident)
    # print "angle", angle, "dist", dist
    # if angle > 30 or dist > 3:
    #     print "update_new_track_window"
    #
    #     new_track_window = move_track_window(new_track_window, old_graident)
    #
    # return new_track_window


def get_gradient_length(gradient):
    return np.linalg.norm(np.array(gradient))


def get_distance(old_track_window, new_track_window):
    old_centroid = np.array(get_mapped_centroid(old_track_window))
    new_centroid = np.array(get_mapped_centroid(new_track_window))
    dist = np.linalg.norm(new_centroid - old_centroid)
    return dist


def move_track_window(track_window, velocity):
    x,y,w,h = track_window
    delta_x, delta_y = velocity
    return (x + delta_x, y + delta_y, w, h)


def get_gradient(w1, w2):
    x1, y1 = get_mapped_centroid(w1)
    x2, y2 = get_mapped_centroid(w2)
    return (x2 - x1, y2 - y1)


def get_color_filtered_frame(frame, player):
    print frame.shape
    print court_mask.shape
    court_filtered_frame = cv2.bitwise_and(frame, court_mask)

    mask_lower_bound = np.array(player.mask_lower_bound, dtype = "uint8")

    mask_upper_bound = np.array(player.mask_upper_bound, dtype = "uint8")

    color_mask = cv2.inRange(frame, mask_lower_bound, mask_upper_bound)

    color_filtered_frame = cv2.bitwise_and(court_filtered_frame, court_filtered_frame, mask=color_mask)

    return cv2.cvtColor(color_filtered_frame, cv2.COLOR_BGR2GRAY)


def update_track_window(frame, player):
    color_filtered_frame = get_color_filtered_frame(frame, player)

    old_track_window = player.get_current_track_window()

    new_track_window = mean_shift_tracking_window(color_filtered_frame, old_track_window, 10)

    # if len(player.get_track_windows()) == player.total_track_window_size:
    #     new_track_window = justified_track_window(player, new_track_window)

    player.set_temp_new_track_window(new_track_window)



def setup_all_track_windows():
    red_track_windows = [
		(2028, 154, 20, 36),
		(2097, 110, 17, 30),
		(2228, 131, 17, 31),
		(2267, 157, 21, 37),
		(2298, 243, 27, 48),
		(2355, 136, 20, 35),
		(2395, 115, 16, 28),
		(2709, 211, 24, 43)
    ]

    blue_track_windows = [
		(2062, 171, 19, 33),
		(2245, 196, 22, 40),
		(2307, 126, 16, 29),
		(2366, 150, 16, 29),
		(2401, 138, 20, 35),
		(2430, 126, 19, 27),
		(2430, 151, 17, 31),
		(2506, 145, 19, 33),
		(2561, 157, 22, 39),
		(2691, 192, 23, 41)
    ]

    yellow_track_windows = [
        (2300, 154, 20, 36)
    ]
    # red_gradients = []
    # for i in range(len(red_track_windows)):
    #     red_gradients.append((0, 0))
    #
    # blue_gradients = []
    # for i in range(len(blue_track_windows)):
    #     blue_gradients.append((0, 0))


    return (red_track_windows, blue_track_windows, yellow_track_windows)


def read_frame(frame_num):
    return cv2.imread("output_frames/fr_{}.jpg".format(frame_num))


def unit_vector(vector):
    return np.array(vector) / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = np.array(unit_vector(v1))
    v2_u = np.array(unit_vector(v2))

    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0
        else:
            return 180
    return abs(int(angle * 180 / np.pi))
#
# def get_background(video_name):
#     cap = cv2.VideoCapture(video_name)
#
#     # frame_height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
#     # frame_width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
#     frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
#
#     ret, first_frame = cap.read()
#     frame_sum = np.float32(first_frame)
#
#     for fr in range(1, frame_count):
#         ret, frame = cap.read()
#         frame_sum += np.float32(frame)
#
#     average_frame = frame_sum / frame_count
#
#     return cv2.convertScaleAbs(average_frame)

main()