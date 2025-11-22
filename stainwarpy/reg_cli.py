import typer
import os
import numpy as np
import json
from tifffile import imwrite
from skimage.transform import AffineTransform, resize
from .regPipeline import registration_pipeline
from .preprocess import extract_channel, load_image_data
from .reg import transform_seg_mask


app = typer.Typer(help="Register H&E stained images to multiplexed images using a feature based registration pipeline.")


@app.command(name="register")
def register(
    fixed_path: str = typer.Argument(..., help="Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    moving_path: str = typer.Argument(..., help="Path to the moving image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder: str = typer.Argument(..., help="Folder to save the registered images and metrics"),
    fixed_img: str = typer.Argument(..., help="Type of fixed image: ['multiplexed', 'hne']"),
    fixed_px_sz: float = typer.Option(None, help="Pixel size of the fixed image (if image is not .ome.tif)"),
    moving_px_sz: float = typer.Option(None, help="Pixel size of the moving image (if image is not .ome.tif)"),
    feature_tform: str = typer.Option('similarity', help="Feature transformation method ['similarity', 'affine', 'projective']. 'similarity' by default and recommended.", show_default=True)
):
    # run the pipeline
    transformation_map, final_img, tre, mi = registration_pipeline(
        fixed_path,
        moving_path,
        fixed_px_sz,
        moving_px_sz,
        fixed_img,
        feature_tform=feature_tform
    )

    output_folder_path = os.path.join(output_folder, "results")
    os.makedirs(output_folder, exist_ok=True)

    # save registration metrics
    metrics_output_path = os.path.join(output_folder_path, "registration_metrics.json")

    with open(metrics_output_path, "w") as f:
        json.dump({"TRE": tre, "Mutual Information": mi}, f)
    print(f"Registration metrics saved to {metrics_output_path}")

    # save registered image
    final_img_path = os.path.join(output_folder_path, "0_final_channel_image.tif")
    imwrite(final_img_path, final_img)

    print(f"Registered image saved to {final_img_path}")

    # save transformation map

    np.save(os.path.join(output_folder_path, f"feature_based_transformation_map.npy"), transformation_map.params)

    print(f"Transformation maps saved to {output_folder_path}/feature_based_transformation_map.npy")



@app.command(name="extract-channel")
def extract_channel_cmd(
    file_path: str = typer.Argument(..., help="Path to the input image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the image with extracted channel"),
    channel_idx: int = typer.Option(0, help="Channel index to extract (Default = 0 for DAPI)", show_default=True),
):
    img = load_image_data(file_path)
    img_ch = extract_channel(img, channel_idx)

    img_folder_path = os.path.join(output_folder_path, "channel_extracted_image")
    os.makedirs(img_folder_path, exist_ok=True)
    img_path = os.path.join(img_folder_path, f"multiplexed_channel_{channel_idx}.tif")
    imwrite(img_path, img_ch)
    print(f"Image with extracted channel saved to {img_path}")



@app.command(name="transform-seg-mask")
def transform_seg_mask_cmd(
    mask_path: str = typer.Argument(..., help="Path to the segmentation mask of the moving image (.npy)"),
    fixed_path: str = typer.Argument(..., help="Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the transformed segmentation mask"),
    tform_map_path: str = typer.Argument(..., help="Path to the transformation map"),
    moving_px_sz: float = typer.Argument(..., help="Pixel size of the moving image"),
    fixed_px_sz: float = typer.Option(None, help="Pixel size of the fixed image (if image is not .ome.tif)", show_default=True)
):
    # load mask
    mask = np.load(mask_path) # will need to change according to mask format
    print(f"Loaded segmentation mask.")

    # load and create transformation parameter objects
    transformation_maps= AffineTransform(matrix=np.load(os.path.join(tform_map_path)))

    print("Loaded transformation map.")

    fixed_init = load_image_data(fixed_path)

    if fixed_px_sz is None:
        try:
            fixed_px_sz, _ = get_pixel_size_ome_tiff(fixed_path)
        except Exception:
            fixed_px_sz = None
        
        if fixed_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for fixed image. Please provide fixed_px_sz.")
    
    scale = moving_px_sz / fixed_px_sz
    if len(fixed_init.shape) == 2:
        fixed_init_sc = resize(fixed_init, (int(fixed_init.shape[0]/scale), int(fixed_init.shape[1]/scale)), anti_aliasing=True)
    else:
        fixed_init_sc = resize(fixed_init, (int(fixed_init[:, :, 0].shape[0]/scale), int(fixed_init[:, :, 0].shape[1]/scale)), anti_aliasing=True)
    fixed_img_shape = (int(fixed_init_sc.shape[0]), int(fixed_init_sc.shape[1]))

    moved_mask = transform_seg_mask(mask, transformation_maps, output_shape=fixed_img_shape)

    os.makedirs(output_folder_path, exist_ok=True)
    np.save(os.path.join(output_folder_path, "transformed_segmentation_mask.npy"), moved_mask)
    print(f"Transformed segmentation mask saved to {output_folder_path}/transformed_segmentation_mask.npy")



def main():
    app()


if __name__ == "__main__":
    main()