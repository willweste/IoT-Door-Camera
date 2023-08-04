# video_wiper.py

import time
import datetime
import os

print("Deletion will commence")

# Directory for videos
video_directory = '/Users/willweste/Door-Camera-Videos'
# Number of days
dayLimit = 5

def clean_up_old_videos():
    # loop through files in directory
    for f in os.listdir(video_directory):
        # Split file name at .
        vid = f.split(".")
        # Only get the name and exclude the file type
        vidFileName = vid[0]
        # Turn the filename into a date object
        date = vidFileName
        date = datetime.datetime.strptime(date, "%d-%m-%Y-%H-%M-%S")
        # Subtract current date from date
        intDate = (datetime.datetime.now() - date).days
        if intDate > dayLimit:
            os.remove(os.path.join(video_directory, f))


# Call the cleanup function
clean_up_old_videos()
