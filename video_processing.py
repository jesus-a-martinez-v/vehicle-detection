from moviepy.editor import VideoFileClip
from image_processing import process_image
from data import load_database

__db = load_database()


def process_video(video_path, output_path='vehicle_project_video_out.mp4'):
    """
    Takes an input video saves a new video with cars properly detected.
    :param video_path: Input video location.
    :param output_path: Output video location.
    """
    # Load the original video
    input_video = VideoFileClip(video_path)

    # Process and save
    def process(i):
        return process_image(img=i, x_start=700, y_start=400, y_stop=656, scales=__db['parameters']['scales'],
                             classifier=__db['model'], scaler=__db['scaler'], parameters=__db['parameters'],
                             heatmap_threshold=5)

    processed_video = input_video.fl_image(process)
    processed_video.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    input_video_path = 'project_video.mp4'
    process_video(input_video_path)
