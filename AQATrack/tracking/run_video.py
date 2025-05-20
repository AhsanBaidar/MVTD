from lib.test.evaluation.tracker import Tracker

# Directly set your parameters here
tracker_name = 'aqatrack'  # Replace with your tracker name
tracker_param = 'AQATrack-ep100-got-256'  # Replace with your parameter file name
videofile = '/home/mbzirc/Downloads/AhsanBB/Dataset_Paper/Codes/pytracking_New/pytracking/Challenge_Videos/IMG_3036.MOV'  # Replace with the path to your video file
#optional_box = [253.0,261.53,28.4,11.41]  # Set to [x, y, w, h] if you have an optional bounding box, otherwise None
optional_box=[606,496,348,178]
debug = None  # Set debug level if needed, otherwise None
save_results = True  # Set to True if you want to save the results
save_name='output'
def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your video file."""
    tracker = Tracker(tracker_name, tracker_param,'got10k_test')
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)

def main():
    run_video(tracker_name, tracker_param, videofile, optional_box, debug, save_results)

if __name__ == '__main__':
    main()



