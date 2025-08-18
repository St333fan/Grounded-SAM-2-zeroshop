import glob
import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# Set the dataset root directory
dataset_root = "/home/appuser/Grounded-SAM-2/dataset/"

# Utility: Extract 20 frames from a video and save as PNGs, return list of 20 .jpg names for loading
def extract_and_prepare_frames_from_video(video_folder, out_folder="train_pbr/000000/rgb", n_frames=20):
    os.makedirs(out_folder, exist_ok=True)
    # Find the first video file in the folder (mp4, avi, mov, mkv)
    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):  # add more if needed
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {video_folder}")
    video_path = video_files[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Always use the first frame, then sample the rest evenly
    frame_indices = [0]
    if n_frames > 1:
        step = (total_frames - 1) / (n_frames - 1)
        frame_indices += [int(round(i * step)) for i in range(1, n_frames)]
    frame_indices = sorted(set(min(idx, total_frames - 1) for idx in frame_indices))
    saved_png_paths = []
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_folder, f"{i:06d}.png")
        cv2.imwrite(out_path, frame)
        saved_png_paths.append(out_path)
    cap.release()
    # For loading in this script: create 20 .jpg file names from 000001.jpg to 000020.jpg (not saved)
    jpg_names = [f"{i+1:06d}.jpg" for i in range(n_frames)]
    return saved_png_paths, jpg_names # non is needed therfore it can be removed in the future


"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-base" # tiny
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# Find all obj_000XXX directories
obj_dirs = [d for d in os.listdir(dataset_root) if d.startswith("obj_") and os.path.isdir(os.path.join(dataset_root, d))]
obj_dirs.sort()

# Loop through each object directory
for obj_dir in obj_dirs:
    obj_path = os.path.join(dataset_root, obj_dir)
    print(f"Processing {obj_path}")

    # Re-initialize models for each object (if needed)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Set up text prompt
    text = "object in the middle."


    # Extract 20 frames from the video in this object directory and save as PNGs
    video_input_folder = obj_path
    rgb_out_folder = os.path.join(obj_path, "train_pbr/000000/rgb")
    saved_png_paths, jpg_names = extract_and_prepare_frames_from_video(video_input_folder, out_folder=rgb_out_folder)

    # Save the same frames as .jpg in sam2frames directory, named 000001.jpg, 000002.jpg, ...
    video_dir = os.path.join(obj_path, "sam2frames")
    os.makedirs(video_dir, exist_ok=True)
    frame_names = []
    for i, png_path in enumerate(saved_png_paths):
        jpg_name = f"{i+1:06d}.jpg"
        jpg_path = os.path.join(video_dir, jpg_name)
        img = cv2.imread(png_path)
        if img is not None:
            cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_names.append(jpg_name)
        else:
            print(f"Warning: Could not read {png_path} for conversion to JPG.")

    if not frame_names:
        print(f"No frames found in {video_dir}, skipping {obj_dir}")
        continue

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # --- Step 2 and onward: process as before, but inside the loop ---
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        # box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    PROMPT_TYPE_FOR_VIDEO = "box" # or "point"
    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

    # Step 4: Propagate the video predictor to get the segmentation results for each frame
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Step 5: Visualize the segment results across the video and save them
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    for frame_idx, segments in video_segments.items():
        object_ids = list(segments.keys())
        masks = list(segments.values())
        mask_save_dir = os.path.join(obj_path, "train_pbr/000000/mask")
        os.makedirs(mask_save_dir, exist_ok=True)
        for i, mask in enumerate(masks):
            mask_2d = np.squeeze(mask)
            if mask_2d.ndim == 2 and mask_2d.shape[0] > 0 and mask_2d.shape[1] > 0:
                mask_img = (mask_2d * 255).astype(np.uint8)
                mask_filename = f"{frame_idx:06d}_{i:06d}.png"
                cv2.imwrite(os.path.join(mask_save_dir, mask_filename), mask_img)

