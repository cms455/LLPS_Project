import os
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np

def tiffstack2tiff(tiff_stack_path, scale=False):
    # Determine the base path and file name of the input TIFF stack
    base_path = os.path.dirname(tiff_stack_path)
    file_name = os.path.basename(tiff_stack_path)
    name_without_ext = os.path.splitext(file_name)[0]

    # Create the output directory
    output_dir = os.path.join(base_path, name_without_ext)
    os.makedirs(output_dir, exist_ok=True)

    # Load the TIFF stack
    with Image.open(tiff_stack_path) as img:
        # Extract and save each frame
        for i in tqdm(range(img.n_frames)):
            img.seek(i)
            if scale:
                # Convert image to numpy array for scaling
                img_array = np.array(img)
                # Scale the image to span the full 0-255 range
                scaled_img_array = (
                            (img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype(
                    np.uint8)
                # Convert back to PIL Image to save
                img_to_save = Image.fromarray(scaled_img_array)
            else:
                # If not scaling, save the image as is
                img_to_save = img
            output_file_path = os.path.join(output_dir, f'f{i:04d}.tif')
            img_to_save.save(output_file_path)
    print(f'Saved {i+1} tiffs at {output_dir}')

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Convert a TIFF stack to individual TIFF files.')
    parser.add_argument('tiff_stack_path', type=str, help='Path to the TIFF stack file.')
    parser.add_argument('-s', '--scale', help='Scale a color range (Option for a tiffstack)', type=bool, default=False)

    # Parse arguments
    args = parser.parse_args()

    # Run the function
    tiffstack2tiff(args.tiff_stack_path, scale=args.scale)
