# For wiping old videos off of the storage
import time
import datetime
import os

# Directory for videos
video_directory = '/Users/willweste/Door-Camera-Videos'

# Number of days
dayLimit = 5

# Current time in the date format videos are stored
current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

# loop through files in directory
for f in os.listdir(video_directory):
    # Split file name at .
    vid = f.split(".")
    # Only get the name and exclude the file type
    vidFIleName = vid[0]
    # Turn the filename into a date object
    date = ""
    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    date = vidFIleName
    date = datetime.datetime.strptime(date, "%d-%m-%Y-%H-%M-%S")
    # Subtract current date from date
    intDate = (datetime.datetime.now() - date).days
    if(intDate > dayLimit):
        os.remove("/Users/willweste/Door-Camera-Videos/"+f)



