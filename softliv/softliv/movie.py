import subprocess
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import cv2
import re
import argparse
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm
from tiffstack2tiff import tiffstack2tiff

'''
Module with functions for making movies
- make_movie: Make a movie from a sequence of images using ffmpeg
- addTimeStamp: Add a timestamp to a video
- addScaleBar: Add a scale bar to a video
- applyLUT: Apply a look-up table to a video
- addText: Add text to an image

Examples:
    # Make a movie from a directory of images
        python .\movie.py -i /path/to/img_dir
        
    # Make a movie from a tiffstack
        python .\movie.py -i /path/to/tiffstack.tif
        
    # Save a movie from a directory of images at a specified location
        python .\movie.py -i /path/to/img_dir -o /path/to/movie.mp4
        
    # Overwrite a movie
        python .\movie.py -i /path/to/img_dir -w
        
    # Add a timestamp to a movie
    ## t toggles the timestamp.
        python .\movie.py -i /path/to/movie.mp4 -t -dt 1.003 # dt is the time interval between frames in seconds
        
    # Add a scale bar to a movie
    ## s toggles the scale bar.
    ## sl is the length of the scale bar in micrometers
    ## scale is the conversion factor from micrometers to pixels (micrometers per pixel).
        python .\movie.py -i /path/to/movie.mp4 -s -sl 10 -scale 0.2 # sl is the length of the scale bar in micrometers
    
    # Color a grayscale movie using a colormap
    ## c toggles the color
    ## cmap is the name of the colormap. 
       For choices of colormaps, see https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
    ## Threshold is the relative threshold for the colormap (0-1). It is the maximum value of the colormap.
        python .\movie.py -i /path/to/movie.mp4 -c -cmap viridis -threshold 1
    
    
    # Add a timestamp, a scale bar, and color to a movie
        python .\movie.py -i -i /path/to/img_dir -s -sl 20 -t -dt 1.000 -f 50 -o /path/to/movie.mp4

'''

# Abs path to ffmpeg
path_mod = os.path.abspath(__file__)
mod_dir_path = os.path.dirname(path_mod)
ffmpeg_path = os.path.join(mod_dir_path, 'ffmpeg')
ffmpeg_path_try = 'ffmpeg'  # Windows may not consider the path to ffpeg as an executable. In that case, use 'ffmpeg' instead.

# Determine the OS
isWindows = os.name == 'nt' # Is the OS Windows? Used to enable the glob feature for Windows users.


if os.name == 'nt': # Windows
    # font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Update the path to your font
    font_path = r'C:\Windows\Fonts\arial.ttf'
elif os.name == 'posix': # Linux
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Update the path to your font
else: # Mac
    font_path = '~/Library/Fonts/Arial.ttf'

if not os.path.exists(font_path):
    message = f"The font file does not exist at {font_path}.\nPlease update the path to your font. For Linux, try 'find ~ -name fonts.'"
    raise FileNotFoundError(message)
_fontSize = 16
_font = ImageFont.truetype(font_path, _fontSize)
_fontScale = 1

# OpenCV font
_fonts_cv = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, # 0-2
          cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, # 3-5
          cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC] # 6,7, 16
_font_cv = 2 # 0, ok,
_lineType = cv2.LINE_AA # This is the best option for cv2.putText. It is anti-aliased.
_color = (0, 0, 0) # white

# OpenCV Colormap
cmap_dict = {
    'autumn': cv2.COLORMAP_AUTUMN,
    'bone': cv2.COLORMAP_BONE,
    'jet': cv2.COLORMAP_JET,
    'winter': cv2.COLORMAP_WINTER,
    'rainbow': cv2.COLORMAP_RAINBOW,
    'ocean': cv2.COLORMAP_OCEAN,
    'summer': cv2.COLORMAP_SUMMER,
    'spring': cv2.COLORMAP_SPRING,
    'cool': cv2.COLORMAP_COOL,
    'hsv': cv2.COLORMAP_HSV,
    'pink': cv2.COLORMAP_PINK,
    'hot': cv2.COLORMAP_HOT,
    'parula': cv2.COLORMAP_PARULA,
    'magma': cv2.COLORMAP_MAGMA,
    'inferno': cv2.COLORMAP_INFERNO,
    'plasma': cv2.COLORMAP_PLASMA,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'cividis': cv2.COLORMAP_CIVIDIS,
    'twilight': cv2.COLORMAP_TWILIGHT,
    'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
    'turbo': cv2.COLORMAP_TURBO,
    'deepgreen': cv2.COLORMAP_DEEPGREEN
}
cmapname_dict = {value: key for key, value in cmap_dict.items()}


def is_tiff_stack(filename):
    """Check if a file is a TIFF stack by attempting to move to the second frame."""
    try:
        with Image.open(filename) as img:
            # Attempt to move to the second frame
            img.seek(1)
    except EOFError:
        # If EOFError is caught, then the file has only one frame
        return False
    except Exception as e:
        # If any other error occurs, print it and indicate uncertainty about the file being a TIFF stack
        print(f"An error occurred: {e}")
        return False
    else:
        # If no error occurs, the file has more than one frame
        return True

def detect_color_space(frame):
    """
    Detect the color space of an image

    Parameters
    ----------
    frame: np.ndarray, image

    Returns
    -------
    color_space: str
    """
    if len(frame.shape) == 2:
        return "Grayscale"
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        return "RGB"
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        return "RGBA"
    else:
        return "Unknown"

def calculate_fontScale(source, relative_fontsize=0.1, fontSize_default=_fontSize):
    """
    Calculate the fontScale for the timestamp based on the size of the video frame
    Parameters
    ----------
    source: str, path to the video

    Returns
    -------
    fontScale: float
    """
    cap = cv2.VideoCapture(source)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fontSize_want = frame_size[1] * relative_fontsize
    fontScale = fontSize_want / fontSize_default # desired font size / default font size
    cap.release()
    cv2.destroyAllWindows()
    return fontScale

def get_img_dimensions(source):
    """
    Get the dimensions of the video frame

    Parameters
    ----------
    source: str, path to the video

    Returns
    -------
    width, height: int, int
    """
    cap = cv2.VideoCapture(source)
    width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    cv2.destroyAllWindows()
    return width, height


def make_movie(source, movname=None, indexsz='05', framerate=10, rm_images=False,
               save_into_subdir=False, start_number=0, framestep=1, ext='png', option='normal', overwrite=False,
               invert=False, add_commands=[]):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    ... ffmpeg is not smart enough to recognize a pattern like 0, 50, 100, 150... etc.
        It tries up to an interval of 4. So 0, 3, 6, 9 would work, but this hinders practicality.
        Use the glob feature in that case. i.e. option='glob'

    Parameters
    ----------
    imgname : str
        ... path and filename for the images to turn into a movie
        ... could be a name of directory where images are stored if option is 'glob'
    movname : str
        path and filename for output movie (movie name)
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None
    option: str
        If "glob", it globs all images with the extention in the directory.
        Therefore, the images does not have to be numbered.
    add_commands: list
        A list to add extra commands for ffmpeg. The list will be added before output name
        i.e. ffmpeg -i images command add_commands movie_name
        exmaple: add_commands=['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']
    """

    if os.path.isdir(source):
        imgdir = Path(source)
        if option == 'glob':
            # input = imgdir.glob(f'*.{ext}')
            input = imgdir / f'*.{ext}'
        else:
            # input = imgdir.glob(f'*%{indexsz}.{ext}')
            input = imgdir / f'*%{indexsz}.{ext}'
    else:
        input = imgname = Path(source)
    # if movie name is not given, name it as same as the name of the img directory
    if movname is None:
        movname = source
        # Name the movie file as the name of the image directory
        if os.path.isdir(source):
            dest = movname = str(Path(source)) + '.mp4'
        # Or, name the movie file as the name of the image file + .mp4
        else:
            dest = movname = Path(source).parent / (Path(source).stem + '.mp4')
    else:
        if os.path.splitext(movname)[1] == '':
            movname = movname + '.mp4'
        dest = movname = Path(movname)

    input = input.__str__() # make it a string
    dest = dest.__str__() # make it a string
    movname = movname.__str__() # make it a string

    if option == 'glob' and os.path.isdir(source):
        if not isWindows:
            # If images are not numbered or not labeled in a sequence, you can use the glob feature.
            # On command line,
            # ffmpeg -r 1
            # -pattern_type glob
            # -i '~/Documents/git/takumi/library/image_processing/images2/*.png'  ## It is CRITICAL to include '' on the command line!
            # -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/Documents/git/takumi/library/image_processing/images2/sample.mp4
            command = [ffmpeg_path,
                     '-pattern_type', 'glob',  # Use glob feature
                     '-r', str(int(framerate)),  # framerate
                     '-i', input,  # images
                     '-vcodec', 'libx264',  # codec
                     '-crf', '12',  # quality
                     '-pix_fmt', 'yuv420p',
                       ]
        else:
            print("... Windows does not support the glob feature. Using the 'concat demuxer' instead.")
            # Concat demuxer: https://trac.ffmpeg.org/wiki/Slideshow
            contents = imgdir.glob(f'*.{ext}')
            contents = [c.__str__() for c in contents]
            # Natural-sort the contents
            contents = natural_sort(contents)
            input = Path(source).parent / 'input.txt'
            with open(input, 'w') as f:
                for content in contents:
                    f.write(f"file '{content}'\n")
                    f.write(f"duration 1\n")
                # For a bug, you must write the last file twice.
                f.write(f"file '{content}'\n")

            command = [ffmpeg_path,
                     '-r', str(int(framerate)),  # framerate
                     '-f', 'concat', # concat demuxer
                     '-safe', '0', # for Windows
                     '-i', input.__str__(),  # text file with the list of paths to the images
                     '-vcodec', 'libx264',  # codec
                     '-crf', '12',  # quality
                     '-pix_fmt', 'yuv420p',
                    ]

    else:
        command = [ffmpeg_path,
                   '-r', str(int(framerate)),
                   '-start_number', str(start_number),
                   '-i', input,  # Input
                   '-pix_fmt', 'yuv420p',
                   '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100']
    if overwrite:
        command.append('-y')
    if invert: # invert colors
        command.append('-vf')
        command.append('negate')
    # check if image has dimensions divisibly by 2 (otherwise, ffmpeg raises an error.)
    # ffmpeg raises an error if image has dimension indivisible by 2. Always make sure that this is not the case.
    # image_paths = glob.glob(imgname + '/*.' + ext)
    # img = mpimg.imread(image_paths[0])
    # height, width = img.shape
    # if not (height % 2 == 0 and width % 2 == 0):
    command += ['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']
    command += add_commands
    command.append(dest)

    # Finally, show the command
    print('\n\n------------------------------')
    print('--------FFMPEG COMMAND--------')
    print('------------------------------')
    print(' '.join(command))
    print('------------------------------')
    print('------------------------------')

    # Execute the command
    try:
        subprocess.call(command)
    except:
        command[0] = ffmpeg_path_try
        subprocess.call(command)
    print('\n\n Movie saved as ' + movname + '.\n\n')

    # # Delete the original images
    # if rm_images:
    #     print('Deleting the original images...')
    #     if not save_into_subdir and imgdir is None:
    #         imdir = os.path.split(imgname)
    #     print('Deleting folder ' + imgdir)
    #     subprocess.call(['rm', '-r', imgdir])
    return dest


def addText(frame, xy_in_px, text, font=_font, color=_color, **kwargs):
    """

    Parameters
    ----------
    frame: np.ndarray, image
    xy_in_px: tuple, (x, y) in pixels
    text: str, text to add
    font: int, font type
    color: tuple, (r, g, b), 0-255
    kwargs: dict, optional, parameters for ImageDraw.Draw().text()

    Returns
    -------
    frame: np.ndarray, image with the text added
    """
    if isinstance(font, int):
        raise ValueError('font must be a PIL.ImageFont object, not an integer.')

    # Convert to PIL image
    pil_frame = Image.fromarray(frame)

    # Prepare to draw on the image
    draw = ImageDraw.Draw(pil_frame)

    # Add text to the image
    draw.text(xy_in_px, text, font=font, color=color, **kwargs)

    # Convert back to OpenCV image
    frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
    return frame

def addTimeStamp(frame, current_frame, fps_in, fps_out, deltaT=1., xy=(0.05, 0.05), color='w',
                 fontScale=_fontScale, fontThick=1, font=_font, lineType=_lineType, text_bg=True,
                 bg_white=False, bg_alpha=0.5):
    """
    Add timestamp to the frame

    Parameters
    ----------
    frame: np.ndarray
    current_frame
    fps_in: flaot, frame rate of the input video in fps,
    fps_out: float,
    xy: tuple, (x, y) in relative coordinates (0-1, 0-1). The bottom left is (0, 0).
    color
    fontScale
    fontThick
    font
    text_bg
    bg_white
    bg_alpha

    Returns
    -------

    """
    # position of the timestamp
    width, height = np.shape(frame)[1], np.shape(frame)[0]
    xyTimeStamp = int(xy[0] * width), int((1-xy[1]) * height - font.size * 4/3.) # 1 pt = 4/3 px

    fps_raw = 1. / deltaT # Frame rate of the raw video in fps
    time_sec = current_frame * deltaT

    hrs = int(np.floor_divide(time_sec, 3600))
    mins = int(np.floor_divide(np.mod(time_sec, 3600), 60))
    sec = int(np.mod(time_sec, 60))
    millisec = int(np.mod(time_sec, 1) * 1e3)
    speed =  fps_in * deltaT # Speed of the video relative to the real time
    # Format of the timestamp
    ## 1. Speed, Hours, Minutes, Seconds, Milliseconds, Frame Number
    # timestamp = f'{speed:.2f}x {hrs:03}:{mins:02}:{sec:02}.{millisec:03} f{current_frame:06}'
    ## 2. Hours, Minutes, Seconds, Milliseconds
    timestamp = f'{hrs:03}:{mins:02}:{sec:02}.{millisec:03}'

    # Add the timestamp to the frame
    # Background of the text (white or black with alpha)
    # Color of the timestamp (0-255, 0-255, 0-255)
    if isinstance(color, str):
        color = tuple(cname2rgba(color)*255) # (r, g, b), 0-255

    # Insert the timestamp
    if type(font) == int: # If font is an integer, use cv2.putText
        if text_bg:
            text_size, _ = cv2.getTextSize(timestamp, font, fontScale, fontThick)
            text_w, text_h = text_size
            pad = int(text_h * 0.25)

            if bg_white:
                bg_color = (255, 255, 255)  # white
            else:
                bg_color = (0, 0, 0)  # black
            # Location of the text
            x0, y0 = (xyTimeStamp[0], xyTimeStamp[1])
            # Add a background to the text, alpha: opacity
            for c in range(0, 3):
                frame[y0 - text_h - pad:y0 + 2 * pad,
                x0 - pad:x0 + text_w + pad, c] = (
                        bg_alpha * bg_color[c] + (1 - bg_alpha) * frame[y0 - text_h - pad:y0 + 2 * pad,
                                                                  x0 - pad:x0 + text_w + pad, c])
        cv2.putText(frame, timestamp, org=(x0, y0),
                fontFace=font, fontScale=fontScale, thickness=fontThick, color=color, lineType=lineType)
    else:
        frame = addText(frame, xyTimeStamp, timestamp, font=font, color=color)
    return frame

def addTimeStamps(source, dest=None,
                  framerate=1., deltaT=1.,
                  xy=(20, 30),
                  color='white', font=_font, fontScale=_fontScale, fontThick=1, lineType=_lineType,
                  close=True,
                  text_bg=True, bg_white=False, bg_alpha=0.5, overwrite=False):
    """
    Add timestamps to the video.

    INPUT: VIDEO (mp4, avi, etc.)
    OUTPUT: VIDEO with timestamps(mp4, avi, etc.)

    Parameters
    ----------
    source: str, path to the video
    dest: str, path to the output video

    Returns
    -------

    """
    cap = cv2.VideoCapture(source)
    fps_in = int(cap.get(cv2.CAP_PROP_FPS)) # frame rate of the input video (This is not the frame rate of the original footage)
    nframes_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if dest is None:
        dest = os.path.splitext(source)[0] + '_timestamp' + os.path.splitext(source)[1]

    fps_out = fps_in
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(dest, fourcc, fps_out, frame_size, isColor=True)

    # print(fps_in, fps_out)
    for current_frame in tqdm(range(nframes_total)):
        ret, frame = cap.read()
        frame = addTimeStamp(frame,
                      current_frame,
                      fps_in,
                      fps_out,
                      deltaT=deltaT,
                      xy=xy,
                      color=color,
                      font=font,
                      fontScale=fontScale,
                      fontThick=fontThick,
                      lineType=lineType,
                      text_bg=text_bg,
                      bg_white=bg_white,
                      bg_alpha=bg_alpha)

        out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    return dest

def addScaleBar(frame, length_in_um, scale_umpx=660./512., xy=(0.7, 0.8), thickness=10, color=(0,0,0),
                annotate=True, fontScale=_fontScale, fontThickness=1, font=_font, lineType=_lineType,

                ):
    # Get the dimensions of the video frame
    width, height = np.shape(frame)[1], np.shape(frame)[0]
    # Position of a scale bar
    xy_in_px = int(xy[0] * width), int((1-xy[1]) * height - font.size * 4/3.) # 1 pt = 4/3 px

    # Scale bar length in pixels
    length_in_px = length_in_um / scale_umpx

    # Scale bar position
    x1, y1 = xy_in_px
    x2, y2 = int(x1 * width + length_in_px), y1 + thickness

    # If the scale bar goes off the frame, adjust the position
    if x2 > width:
        x2 = int(width * 0.95)
        x1 = int(width * 0.95 - length_in_px)
    if y2 > height:
        y2 = int(height * 0.95)
        y1 = int(height * 0.95 - thickness)

    # Color of the timestamp (0-255, 0-255, 0-255)
    if isinstance(color, str):
        color = tuple(cname2rgba(color) * 255)  # (r, g, b), 0-255

    # Draw the scale bar using cv2.rectangle() method
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1) # -1 thickness fills the rectangle

    if annotate:
        text = f'{length_in_um:.0f} µm'
        xyScaleBarAnnotation = (int(x1), min(height, int(y1 + thickness*1.5)))
        # Right-end position of the annotation
        x2Annotation = xyScaleBarAnnotation[0] + int(font.size * 4/3 * len(text))
        y2Annotation = xyScaleBarAnnotation[1] + int(font.size * 4/3)
        # If the annotation goes off the frame, adjust the position
        # print(xyScaleBarAnnotation[0], int(font.size))
        if x2Annotation > width:
            x1Annotation = width - int(font.size * 4/3 * len(text) / 2)
            xyScaleBarAnnotation = (x1Annotation, min(height, int(y1 + thickness * 2)))
        if y2Annotation > height:
            y1Annotation = width - int(font.size * 4/3 * len(text) / 2)
            xyScaleBarAnnotation = (x1Annotation, min(height, int(y1 + thickness * 2)))

        # image, text, org, font, fontScale, color
        # cv2.putText(frame, f'{length_in_um}',  # Alt + 230 is the unicode for micro
        #             xy,
        #             fontFace=font, fontScale=fontScale, color=color, thickness=fontThickness, lineType=lineType)
        frame = addText(frame, xyScaleBarAnnotation, text, font=font, color=color )

    return frame


def addScaleBars(source, length_in_um, scale_umpx, xy=(0.7, 0.8),
                 dest=None,
                 relative_thickness=0.01, color='white',
                 font=_font, lineType=_lineType, fontScale=_fontScale,
                 close=True,
                 ):
    """
    Add timestamps to the video.

    INPUT: VIDEO (mp4, avi, etc.)
    OUTPUT: VIDEO with timestamps(mp4, avi, etc.)

    Parameters
    ----------
    source: str, path to the video
    dest: str, path to the output video

    Returns
    -------

    """
    cap = cv2.VideoCapture(source)
    # frame rate of the input video (This is not the frame rate of the original footage)
    fps_in = int(cap.get(cv2.CAP_PROP_FPS))
    nframes_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if dest is None:
        dest = os.path.splitext(source)[0] + '_scalebar' + os.path.splitext(source)[1]
    fps_out = fps_in
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(dest, fourcc, fps_out, frame_size, isColor=True)

    # print(fps_in, fps_out)
    thickness = max(int(relative_thickness * frame_size[1]), 1)
    for current_frame in tqdm(range(nframes_total)):
        ret, frame = cap.read()
        frame = addScaleBar(frame, length_in_um, scale_umpx=scale_umpx, xy=xy,
                    thickness=thickness, color=color,
                    font=font, lineType=lineType, fontScale=fontScale)
        out.write(frame)
    cap.release()
    cv2.destroyAllWindows()
    return dest


def applyLUT(source, cmap, threshold=1, dest=None, close=True):
    cap = cv2.VideoCapture(source)
    # frame rate of the input video (This is not the frame rate of the original footage)
    fps_in = int(cap.get(cv2.CAP_PROP_FPS))
    nframes_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if dest is None:
        if isinstance(cmap, str):
            cmapname = cmap
        elif isinstance(cmap, int):
            cmapname = cmapname_dict[cmap]
        else:
            raise ValueError('cmap must be a string or an integer (cv2.COLORMAP_XXX)')
        dest = os.path.splitext(source)[0] + '_' + cmapname + os.path.splitext(source)[1]
    fps_out = fps_in
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(dest, fourcc, fps_out, frame_size, isColor=True)
    for i, current_frame in enumerate(tqdm(range(nframes_total))):
        ret, frame = cap.read()
        if i == 0:
            colorSpace = detect_color_space(frame)
        if colorSpace in ['RGB', 'RGBA']:
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = gray2rgb(frame, cmap, threshold)
        out.write(frame)

    if close:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    return dest


def gray2rgb(frame, cmap=cmap_dict['viridis'], threshold=1, c=4):
    """
    Convert a grayscale image to a color image using a colormap
    Parameters
    ----------
    frame: np.ndarray, grayscale image
    cmap: str or int, colormap
    ... For choices of colormaps, see https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
    threshold: float, threshold for the colormap

    Returns
    -------
    frame: np.ndarray, color image
    """
    if isinstance(cmap, str):
        cmap = cmap_dict[cmap]
    # Normalize the frame
    frame = frame / (frame.max() * c)
    frame[frame > threshold] = threshold
    # Apply the colormap
    frame = cv2.applyColorMap((frame * 255).astype(np.uint8), cmap)
    return frame

def natural_sort(arr):
    """
    natural-sorts elements in a given array
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    e.g.-  arr = ['a28', 'a01', 'a100', 'a5']
    ... WITHOUT natural sorting,
     -> ['a01', 'a100', 'a28', 'a5']
    ... WITH natural sorting,
     -> ['a01', 'a5', 'a28', 'a100']


    Parameters
    ----------
    arr: list or numpy array of strings

    Returns
    -------
    sorted_array: natural-sorted

    """

    def atoi(text):
        'natural sorting'
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    return sorted(arr, key=natural_keys)

def hex2rgb(hex, normalize=True):
    """
    Converts a HEX code to RGB in a numpy array
    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    h = hex.strip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    if normalize:
        rgb = rgb / 255.

    return rgb


def cname2hex(cname):
    """
    Converts a color registered on matplotlib to a HEX code
    Parameters
    ----------
    cname

    Returns
    -------

    """
    try:
        hex = mcolors.CSS4_COLORS[cname]
        return hex
    except NameError:
        print(cname, ' is not registered as default colors by matplotlib!')
        return None


def cname2rgba(cname, normalize=True):
    """
    Converts a color registered on matplotlib to a RGBA code
    Parameters
    ----------
    cname
    normalize

    Returns
    -------

    """
    hex = cname2hex(cname)
    rgba = hex2rgb(hex, normalize=normalize)
    return rgba


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make an avi using images in a specified directory using ffmpeg')
    parser.add_argument('-i', '--source', help='Directory where images are stored', type=str, default=None)
    parser.add_argument('-o', '--dest', help='Directory where movie is exported', type=str, default=None)
    parser.add_argument('-e', '--imgtype', help='Extension of images in the directory. default: png', type=str, default='tif')
    parser.add_argument('-f', '--framerate', help='Frame rate, default: 10', type=int, default=10)
    parser.add_argument('-w', '--overwrite', help='Overwrites a movie', action='store_true')
    parser.add_argument('-dt', '--deltaT', help='Time interval between images in seconds', type=float, default=1.)
    parser.add_argument('-t', '--addTimeStamp', help='Add a time stamp', action='store_true')
    parser.add_argument('-c', '--contrast', help='Scale a color range (Option for a tiffstack)',action='store_true')
    parser.add_argument('-s', '--addScaleBar', help='Add a scalebar', action='store_true')
    parser.add_argument('-sl', '--scale_bar_length', help='Length of a scale bar in micrometers', type=float, default=20.)
    parser.add_argument('-scale', '--scale_umpx', help='Scale in the units of um per pixel. Default: 0.1289um/px', type=float, default=66./512.)
    parser.add_argument('-color', '--LUT', help='Apply look-up table (LUT). '
                                                            'Available colormaps include viridis, turbo, deepgreen, pink, etc. '
                                                            'Check out opencv colormaps for more.',
                                    type=str,
                                    default=None)
    parser.add_argument('-fontsize', '--fontsize', help='Relative font size wrp to the height', type=float,
                        default=0.05)
    parser.add_argument('-tpos', '--timeStampPos',
                        help='Relative position of a time stamp. (0-1, 0-1). (0,0) is the bottom left). e.g. -tpos 0.02 0.01',
                        nargs="+", type=float,
                        default=(0.02, 0.01))
    parser.add_argument('-spos', '--scaleBarPos',
                        help='Relative position of a scale bar.(0-1, 0-1). (0,0) is the bottom left) e.g. -tpos 0.7 0.01',
                        nargs="+",type=float,
                        default=(0.7, 0.01))
    args = parser.parse_args()

    if is_tiff_stack(args.source):
        print('... Converting a TIFF stack to individual TIFF files...')
        tiffstack2tiff(args.source, scale=args.contrast)
        args.source = os.path.splitext(args.source)[0]
    print('... Making a movie with the glob option using natural sorting...')
    output = make_movie(args.source, framerate=args.framerate, movname=args.dest,
                      rm_images=False, ext=args.imgtype, option='glob', overwrite=args.overwrite)
    # Get the dimensions of the video frame
    width, height = get_img_dimensions(output)

    # Configure the font size
    _fontScale = calculate_fontScale(output, relative_fontsize=args.fontsize, fontSize_default=_fontSize)
    _fontSize = int(_fontSize * _fontScale)
    _fontSize_in_px = _fontSize * 4/3 # 1 pt = 4/3 px
    _font = ImageFont.truetype(font_path, _fontSize)

    if args.LUT is not None:
        if args.LUT not in cmap_dict.keys():
            print('... The LUT is not recognized. Please choose from the following list:')
            print(cmap_dict.keys())
            sys.exit()
        output = applyLUT(output, cmap=args.LUT, threshold=0.8)

    if args.addScaleBar:
        output = addScaleBars(output, length_in_um=args.scale_bar_length, scale_umpx=args.scale_umpx,
                              xy=args.scaleBarPos, font=_font, color='white', fontScale=_fontScale,
                              )

    if args.addTimeStamp:
        raw_framerate = 1 / args.deltaT
        # xyTimeStamp = int(args.timeStampPos[0] * width), int(args.timeStampPos[1] * height + _fontSize_in_px)
        output = addTimeStamps(output,
                      framerate=raw_framerate*args.framerate, # Frame rate of the output video
                      dest=None, # Location of the output video with timestamps
                      xy=args.timeStampPos, # Relative position of the timestamp (x, y) from the botom left corner
                      color='white',
                      fontScale=_fontScale,
                      fontThick=1,
                      font=_font, # FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_COMPLEX, etc.
                      text_bg=True, # Whether to add a background to the timestamp
                      bg_white=False, # If text_bg is True, choose the color of the background (white or black)
                      bg_alpha=0.5, # If text_bg is True, choose the opacity of the background (0-1)
                      overwrite=False, # If True, overwrite the original video (Almost never use this option!)
                      lineType=_lineType, # Line type of the timestamp
                      )
