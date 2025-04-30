import pymupdf
import os
import sys
import pathlib
import argparse
import io
from PIL import Image
import multiprocessing as mp
import time
import traceback # To print detailed error info from workers
import shutil # To copy files
import re # For parsing pytesseract output

# --- Add Tesseract Imports ---
try:
    import pytesseract # Import pytesseract
    pytesseract_available = True
except ImportError:
    pytesseract = None
    pytesseract_available = False
    # print("Warning: Pytesseract not found. Tesseract text detection will not be available.")
# --- End Tesseract Imports ---

# --- Add OpenCV Imports ---
try:
    import cv2
    import numpy as np
    opencv_available = True
except ImportError:
    cv2 = None
    np = None
    opencv_available = False
    # print("Warning: OpenCV not found. OpenCV text detection will not be available.")
# --- End OpenCV Imports ---

# --- Global variable to hold the OpenCV model in each worker process ---
# This will be initialized by the worker_init function
_opencv_detector_model = None
# --- End Global variable ---

# --- Worker Initialization Function ---
def worker_init(model_path, use_cuda):
    """
    Initializes the OpenCV text detection model in each worker process.
    This function is called once per process when the pool starts.
    """
    global _opencv_detector_model
    if not opencv_available:
        # print(f"[{mp.current_process().name}] OpenCV library not available in worker. Cannot load model.")
        _opencv_detector_model = None # Ensure it's None if OpenCV isn't there
        return

    if not os.path.exists(model_path):
        print(f"[{mp.current_process().name}] Error: OpenCV model file not found at {model_path}. Text detection will be disabled for this worker.")
        _opencv_detector_model = None # Indicate failure to load
        return

    try:
        # print(f"[{mp.current_process().name}] Loading OpenCV text detection model from {model_path}...")
        model = cv2.dnn.TextDetectionModel_DB(model_path)

        # --- Set CUDA Backend/Target if requested and available ---
        if use_cuda:
            try:
                # Check if CUDA is available in this OpenCV build
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    # print(f"[{mp.current_process().name}] Attempting to use CUDA backend for OpenCV DNN.")
                    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # Or cv2.dnn.DNN_TARGET_CUDA_FP16 for FP16 inference
                    # print(f"[{mp.current_process().name}] OpenCV DNN backend set to CUDA.")
                else:
                    print(f"[{mp.current_process().name}] Warning: CUDA requested but not available in OpenCV build or system. Using CPU.")
                    # Fallback to CPU (which is the default, but explicit is fine)
                    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as cuda_e:
                 print(f"[{mp.current_process().name}] Warning: Failed to set CUDA backend for OpenCV DNN: {cuda_e}. Using CPU.")
                 # traceback.print_exc() # Uncomment for detailed error
                 # Ensure CPU is used if setting CUDA failed
                 model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                 model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
             # print(f"[{mp.current_process().name}] CUDA not requested for OpenCV DNN. Using CPU.")
             # Explicitly set to CPU if not using CUDA (optional, as it's default)
             model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
             model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # --- End CUDA Setup ---


        # Configure model input parameters (using values from reference)
        # These values might need tuning depending on the specific model and image types
        # These parameters are set regardless of backend (CPU or CUDA)
        model.setInputSize(736, 736) # Input size for the model
        model.setInputMean((122.67891434, 116.66876762, 104.00698793)) # Mean subtraction
        model.setInputScale(1.0 / 255.0) # Scale factor
        model.setInputSwapRB(True) # Swap R and B channels

        # Set DB-specific parameters (using values from reference)
        model.setBinaryThreshold(0.3)
        model.setPolygonThreshold(0.5)
        model.setMaxCandidates(200)
        model.setUnclipRatio(2.0)

        _opencv_detector_model = model # Store the loaded model in the global variable
        # print(f"[{mp.current_process().name}] OpenCV model loaded successfully.")

    except Exception as e:
        print(f"[{mp.current_process().name}] Error loading OpenCV model from {model_path}: {e}")
        # traceback.print_exc() # Uncomment for detailed error
        _opencv_detector_model = None # Indicate failure to load

# --- End Worker Initialization Function ---


# --- Renamed Function: Find Smallest Text Height using Tesseract ---
def find_smallest_text_height_tesseract(
    pil_img: Image.Image
) -> int | None:
    """
    Uses Tesseract OCR to find the height of the smallest detected word
    in a Pillow image.

    Args:
        pil_img: The input Pillow Image.

    Returns:
        The minimum height of a detected word in pixels, or None if
        no text is found, Tesseract (library or executable) is not available,
        or an error occurs.
    """
    # Check if the pytesseract library was successfully imported
    if not pytesseract_available:
        # print(f"[{mp.current_process().name}]   Pytesseract library not available. Cannot perform Tesseract text analysis.")
        return None

    min_text_height_px = float('inf')
    text_found = False

    try:
        # Use image_to_data to get bounding boxes for words
        # output_type=pytesseract.Output.STRING gives tab-separated string
        # Hardcoded lang='eng', config='' as per original script
        ocr_data_string = pytesseract.image_to_data(
            pil_img,
            output_type=pytesseract.Output.STRING,
            lang='eng', # Hardcoded
            config=''   # Hardcoded
        )

        # Parse the tab-separated string output
        # The format is level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, text
        lines = ocr_data_string.strip().split('\n')
        if len(lines) > 1: # Skip header line
            for line in lines[1:]:
                fields = line.split('\t')
                # Ensure it's a word-level entry (level 5) and has enough fields
                if len(fields) >= 11 and fields[0] == '5':
                    try:
                        # Bbox is left, top, width, height
                        word_height = int(fields[9])
                        if word_height > 0: # Ignore zero-height boxes
                            min_text_height_px = min(min_text_height_px, word_height)
                            text_found = True
                    except ValueError:
                        # Handle cases where width/height are not integers
                        pass
                    except IndexError:
                         # Handle cases where line doesn't have enough fields unexpectedly
                         pass


    except pytesseract.TesseractNotFoundError:
        # This catches the case where the pytesseract library is installed,
        # but the Tesseract executable is not found in the system's PATH.
        # print(f"[{mp.current_process().name}]   Tesseract executable not found. Cannot perform text analysis.")
        return None # Indicate failure/not found
    except Exception as e:
        # print(f"[{mp.current_process().name}]   Error during Tesseract analysis: {e}")
        # traceback.print_exc() # Uncomment for detailed error
        return None # Indicate failure

    if text_found and min_text_height_px != float('inf') and min_text_height_px > 0:
        return min_text_height_px
    else:
        # print(f"[{mp.current_process().name}]   No usable text found by Tesseract.")
        return None # Indicate no usable text found

# --- New Function: Find Smallest Text Height using OpenCV DB ---
# Modified to accept the pre-loaded model
def find_smallest_text_height_opencv(
    pil_img: Image.Image,
    model # Pass the pre-loaded model
) -> int | None:
    """
    Uses OpenCV's TextDetectionModel_DB to find the height of the smallest
    detected text bounding box in a Pillow image.

    Args:
        pil_img: The input Pillow Image.
        model: The pre-loaded OpenCV TextDetectionModel_DB instance.

    Returns:
        The minimum height of a detected text bounding box in pixels, or None if
        no text is found, OpenCV is not available, the model failed to load,
        or an error occurs.
    """
    # Check if the OpenCV library was successfully imported and model is loaded
    if not opencv_available or model is None:
        # print(f"[{mp.current_process().name}]   OpenCV library or model not available. Cannot perform OpenCV text analysis.")
        return None

    min_text_height_px = float('inf')
    text_found = False

    try:
        # Convert Pillow image to OpenCV format (BGR NumPy array)
        # Ensure image is in a format OpenCV can handle (e.g., RGB or L)
        if pil_img.mode == 'L': # Grayscale
             cv_img = np.array(pil_img)
             # Convert grayscale to BGR as the model expects 3 channels
             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        elif pil_img.mode == 'RGB':
             cv_img = np.array(pil_img)
             # Convert RGB to BGR (OpenCV default)
             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        else:
             # Convert to RGB first if it's a different mode (e.g., RGBA, P)
             cv_img = np.array(pil_img.convert('RGB'))
             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        # Detect text in the image using the pre-loaded model
        # quadrilaterals is a list of numpy arrays, each shape (4, 2)
        quadrilaterals, confidences = model.detect(cv_img)

        if quadrilaterals is not None:
            for quad in quadrilaterals:
                # quad is a numpy array of shape (4, 2) representing the corners
                # Calculate the height of the bounding rectangle for simplicity
                # This gives an axis-aligned bounding box height
                x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
                word_height = h

                if word_height > 0: # Ignore zero-height boxes
                    min_text_height_px = min(min_text_height_px, word_height)
                    text_found = True

    except cv2.error as e:
        # print(f"[{mp.current_process().name}]   OpenCV error during text analysis: {e}")
        # traceback.print_exc() # Uncomment for detailed error
        return None # Indicate failure
    except Exception as e:
        # print(f"[{mp.current_process().name}]   Error during OpenCV text analysis: {e}")
        # traceback.print_exc() # Uncomment for detailed error
        return None # Indicate failure

    if text_found and min_text_height_px != float('inf') and min_text_height_px > 0:
        return min_text_height_px
    else:
        # print(f"[{mp.current_process().name}]   No usable text found by OpenCV.")
        return None # Indicate no usable text found


# --- Custom Image Processing Function ---
# This function takes a Pillow Image and returns a processed Pillow Image.
# It also receives image info and target settings for downsampling/color conversion.
# It now includes logic to find text and adjust resizing based on smallest text height, optionally.
def custom_image_process(
    pil_img: Image.Image,
    img_info: dict,
    target_dpi: int,
    force_color: str,
    text_detection_method: str, # New argument
    min_readable_text_height_px: int
) -> Image.Image:
    """
    Custom image processing, including downsampling, color conversion,
    and optional resizing adjustment based on smallest text size using
    either Tesseract or OpenCV.

    Args:
        pil_img: The input Pillow Image.
        img_info: Dictionary containing information about the image from get_image_info.
                  Includes 'bbox', 'width', 'height', etc.
        target_dpi: The target DPI for downsampling (DPI-based target size).
        force_color: Color conversion setting ('rgb', 'gray', 'none').
        text_detection_method: Method to use for text detection ('tesseract', 'opencv', 'none').
        min_readable_text_height_px: Minimum desired height for the smallest text
                                     in the *resized* image (in pixels).

    Returns:
        The processed Pillow Image.
    """
    original_width = pil_img.width
    original_height = pil_img.height
    bbox = pymupdf.Rect(img_info['bbox']) # Convert bbox tuple to Rect for easier access

    # --- Apply Forced Color Conversion ---
    if force_color == 'rgb' and pil_img.mode != 'RGB':
        # print(f"[{mp.current_process().name}]     Converting image to RGB.")
        pil_img = pil_img.convert("RGB")
    elif force_color == 'gray' and pil_img.mode != 'L':
         # print(f"[{mp.current_process().name}]     Converting image to Grayscale.")
         pil_img = pil_img.convert("L")
    # --- End Forced Color Conversion ---

    # --- Determine Target Dimensions ---
    # Calculate initial target dimensions based on the image's bounding box on the page and target_dpi
    # Bbox dimensions are in points (1 inch = 72 points)
    # Ensure bbox dimensions are positive to avoid issues with zero/negative sizes
    bbox_width_pts = max(1.0, bbox.width)
    bbox_height_pts = max(1.0, bbox.height)

    dpi_based_target_width_px = int(round(bbox_width_pts / 72 * target_dpi))
    dpi_based_target_height_px = int(round(bbox_height_pts / 72 * target_dpi))

    # Avoid making tiny images if calculated dimensions are very small
    min_dim_px = 1 # Minimum dimension in pixels for the final image
    dpi_based_target_width_px = max(dpi_based_target_width_px, min_dim_px)
    dpi_based_target_height_px = max(dpi_based_target_height_px, min_dim_px)

    # Initialize final target size with the DPI-based size
    final_target_width_px = dpi_based_target_width_px
    final_target_height_px = dpi_based_target_height_px

    # --- Optional Text-Based Resizing Adjustment ---
    min_text_height_px_orig = None # Initialize with None

    if text_detection_method != 'none':
        # Call the appropriate text detection function
        if text_detection_method == 'tesseract':
            # print(f"[{mp.current_process().name}]   Performing Tesseract text analysis...")
            min_text_height_px_orig = find_smallest_text_height_tesseract(pil_img)
        elif text_detection_method == 'opencv':
            # print(f"[{mp.current_process().name}]   Performing OpenCV text analysis...")
            # Pass the global model variable to the function
            min_text_height_px_orig = find_smallest_text_height_opencv(pil_img, _opencv_detector_model)
        # else: Should not happen due to argparse choices

        if min_text_height_px_orig is not None: # Check if text was found successfully
            # Calculate the actual height of the smallest text if resized to the DPI-based size
            # Avoid division by zero if original_height is 0
            if original_height > 0:
                actual_text_height_at_dpi_size = min_text_height_px_orig * (dpi_based_target_height_px / original_height)
            else:
                actual_text_height_at_dpi_size = 0 # Cannot resize if original height is 0

            # print(f"[{mp.current_process().name}]   Smallest text height (orig px): {min_text_height_px_orig}")
            # print(f"[{mp.current_process().name}]   Actual text height at DPI size: {actual_text_height_at_dpi_size:.2f} px")
            # print(f"[{mp.current_process().name}]   Min readable text height: {min_readable_text_height_px} px")


            if actual_text_height_at_dpi_size < min_readable_text_height_px:
                # Text is too small at DPI-based size, calculate text-driven target size
                # Calculate the required height for the smallest text to be min_readable_text_height_px
                # Avoid division by zero if min_text_height_px_orig is 0
                if min_text_height_px_orig > 0:
                    required_height_for_text = (min_readable_text_height_px / min_text_height_px_orig) * original_height
                else:
                    required_height_for_text = original_height # Cannot scale based on text, use original height

                # Calculate the corresponding width to maintain aspect ratio
                # Avoid division by zero if original_height is 0
                if original_height > 0:
                    required_width_for_text = required_height_for_text * (original_width / original_height)
                else:
                    required_width_for_text = original_width # Cannot scale based on height, use original width

                # The text-driven target size
                text_driven_target_width = int(round(required_width_for_text))
                text_driven_target_height = int(round(required_height_for_text))

                # Cap the text-driven target size at the original image size (no upscaling beyond original)
                capped_text_driven_width = min(text_driven_target_width, original_width)
                capped_text_driven_height = min(text_driven_target_height, original_height)

                # Ensure capped dimensions are at least 1x1
                capped_text_driven_width = max(1, capped_text_driven_width)
                capped_text_driven_height = max(1, capped_text_driven_height)

                # The final target size is the component-wise maximum of the DPI-based size and the capped text-driven size
                final_target_width_px = max(dpi_based_target_width_px, capped_text_driven_width)
                final_target_height_px = max(dpi_based_target_height_px, capped_text_driven_height)

                # print(f"[{mp.current_process().name}]   Text is too small at DPI size. Required size for text: {capped_text_driven_width}x{capped_text_driven_height}")
                # print(f"[{mp.current_process().name}]   Final target size (max of DPI and Text): {final_target_width_px}x{final_target_height_px}")

            # else: Text is readable at DPI-based size
            # Final target size remains the dpi_based_target_size (initialized earlier)
            # print(f"[{mp.current_process().name}]   Text is readable at DPI size. Using DPI-based target size: {final_target_width_px}x{final_target_height_px}")

        # else: min_text_height_px_orig is None (no text found or error)
        # Final target size remains the dpi_based_target_size (initialized earlier)
        # print(f"[{mp.current_process().name}]   Text analysis skipped or failed. Using DPI-based target size: {final_target_width_px}x{final_target_height_px}")

    # else: text_detection_method is 'none'
    # Final target size remains the dpi_based_target_size (initialized earlier)
    # print(f"[{mp.current_process().name}]   Text-based resize disabled. Using DPI-based target size: {final_target_width_px}x{final_target_height_px}")


    # --- Apply Resizing ---
    # The target size is final_target_width_px x final_target_height_px
    # We cap the target size at original size to prevent upscaling
    # Note: The logic above already caps the text-driven size at original,
    # and the final size is max(DPI_size, capped_text_size).
    # If DPI_size > original, this might still result in upscaling if text_resize is off.
    # Let's ensure the final resize dimensions are capped at original size.
    resize_width = min(final_target_width_px, original_width)
    resize_height = min(final_target_height_px, original_height)


    # Ensure dimensions are at least 1x1 before resizing
    resize_width = max(1, resize_width)
    resize_height = max(1, resize_height)


    if resize_width < original_width or resize_height < original_height:
         # print(f"[{mp.current_process().name}]     Resizing image from {original_width}x{original_height} to {resize_width}x{resize_height}")
         # Use a high-quality resampling filter
         pil_img = pil_img.resize((resize_width, resize_height), Image.Resampling.LANCZOS) # Or BICUBIC, ANTIALIAS, etc.
    # --- End Resizing ---


    # --- Add other custom processing here if needed ---
    # Example: Apply a filter, adjust levels, etc.
    # from PIL import ImageEnhance
    # enhancer = ImageEnhance.Contrast(pil_img)
    # pil_img = enhancer.enhance(1.2) # Increase contrast by 20%
    # --- End other custom processing ---


    return pil_img

# --- Worker Function for Multiprocessing Pool ---
def process_single_pdf(args):
    """
    Processes a single PDF file in a separate process.

    Args:
        args: A tuple containing (absolute_filepath, relative_filepath,
              output_root_dir, errors_dir_name, jpg_quality, enable_font_subsetting,
              target_dpi, force_color, text_detection_method,
              min_readable_text_height_px).
              Note: opencv_model_path and use_cuda are handled by the worker_init function.

    Returns:
        A tuple (relative_filepath, status) where status is 'success' or 'failure'.
    """
    # Unpack arguments
    (absolute_filepath, relative_filepath, output_root_dir, errors_dir_name, jpg_quality, enable_font_subsetting, target_dpi, force_color, text_detection_method, min_readable_text_height_px) = args

    output_filepath = os.path.join(output_root_dir, relative_filepath)
    output_subdir = os.path.dirname(output_filepath)
    errors_filepath = os.path.join(output_root_dir, errors_dir_name, relative_filepath)
    errors_subdir = os.path.dirname(errors_filepath)

    print(f"[{mp.current_process().name}] Processing {relative_filepath}...")

    try:
        doc = pymupdf.open(absolute_filepath)

        # Optional: Handle encrypted files if needed
        # if doc.is_encrypted:
        #     print(f"[{mp.current_process().name}] Skipping encrypted file {relative_filepath} (encrypted).")
        #     doc.close()
        #     return (relative_filepath, 'skipped') # Indicate skipped

        modified = False
        processed_xrefs = set() # Track xrefs processed in this file

        # --- Image Processing Loop ---
        for page in doc:
            image_list = page.get_image_info(xrefs=True)

            for img_info in image_list:
                xref = img_info['xref']
                # bbox = img_info['bbox'] # Bbox is useful for context but not needed for replace_image

                # Process only images referenced by xref (not inline images)
                # and only process each unique image object once within this file
                if xref != 0 and xref not in processed_xrefs:
                    # print(f"[{mp.current_process().name}]   Page {page.number}: Found image with xref {xref}")
                    try:
                        # Create pixmap from the original image object in the PDF
                        orig_pix = pymupdf.Pixmap(doc, xref)

                        # --- Convert PyMuPDF Pixmap to Pillow Image ---
                        pil_img = orig_pix.pil_image()
                        orig_pix = None # Release PyMuPDF pixmap memory

                        # --- Apply Custom Pillow Processing (Downsampling, Color, etc.) ---
                        processed_pil_img = custom_image_process(
                            pil_img,
                            img_info,
                            target_dpi,
                            force_color,
                            text_detection_method, # Pass new argument
                            min_readable_text_height_px
                            # The OpenCV model is accessed via the global _opencv_detector_model inside custom_image_process
                        )
                        pil_img = None # Release original Pillow image memory

                        # --- Convert Processed Pillow Image back to JPEG bytes ---
                        buffer = io.BytesIO()
                        # Save the Pillow image to the BytesIO buffer in JPEG format
                        # Pillow handles colorspace conversion for JPEG saving automatically
                        processed_pil_img.save(buffer, format="jpeg", quality=jpg_quality, optimize=True) # optimize=True is good practice
                        jpeg_bytes = buffer.getvalue()

                        processed_pil_img = None # Release processed Pillow image memory
                        buffer = None # Release buffer memory

                        # Replace the original image object with the new JPEG bytes
                        page.replace_image(xref, stream=jpeg_bytes)
                        # print(f"[{mp.current_process().name}]     Replaced image {xref} with JPEG.")
                        modified = True
                        processed_xrefs.add(xref) # Mark this xref as processed for this file

                    except ImportError:
                         # This should ideally be caught earlier, but as a fallback
                         # This specific ImportError catch might be redundant now with the global flags
                         # but keeping it doesn't hurt.
                         print(f"[{mp.current_process().name}]   Pillow, Tesseract, or OpenCV not installed. Skipping image {xref} on page {page.number}.")
                         # Continue to the next image in this file
                    except Exception as e:
                        print(f"[{mp.current_process().name}]   Error processing image {xref} on page {page.number}: {e}")
                        # traceback.print_exc() # Uncomment for detailed error in worker logs
                        # Continue to the next image in this file
                elif xref == 0:
                     # print(f"[{mp.current_process().name}]   Page {page.number}: Found inline image (xref 0). Skipping replacement.")
                     # Inline images cannot be replaced using page.replace_image(xref, ...)
                     # If you need to process/replace inline images, it requires
                     # parsing and modifying the page's content stream, which is complex.
                     pass # Skip inline images for replacement
        # --- End Image Processing Loop ---

        # --- Apply Font Subsetting (PDF only) ---
        if enable_font_subsetting and doc.is_pdf:
            try:
                # subset_fonts should be called after modifications but before saving with garbage collection
                # print(f"[{mp.current_process().name}]   Applying font subsetting...")
                doc.subset_fonts() # Using native MuPDF subsetting (fallback=False is default)
                modified = True # Font subsetting is a modification
            except Exception as e:
                 print(f"[{mp.current_process().name}]   Error during font subsetting: {e}")
                 # traceback.print_exc() # Uncomment for detailed error in worker logs
        # --- End Font Subsetting ---


        if modified:
            # Ensure output subdirectory exists
            os.makedirs(output_subdir, exist_ok=True)
            # print(f"[{mp.current_process().name}] Saving modified PDF to {output_filepath}...")

            # --- Use save() with maximum compression options ---
            doc.save(
                output_filepath,
                garbage=4,          # Remove unreferenced objects and duplicate streams
                deflate=True,       # Compress all streams
                use_objstms=1,      # Compress objects into streams
                clean=True,         # Optional: Clean content streams (can sometimes help)
                # deflate_images=True, # Included in deflate=True
                # deflate_fonts=True,  # Included in deflate=True
            )
            # --- End save() call ---

            print(f"[{mp.current_process().name}] Successfully processed and saved {relative_filepath}.")
        else:
             # If no changes were made, just copy the original file if output doesn't exist
            if not os.path.exists(output_filepath):
                 # print(f"[{mp.current_process().name}] No images referenced by xref found or replaced, and no font subsetting applied in {relative_filepath}. Copying original.")
                 os.makedirs(output_subdir, exist_ok=True)
                 # Simple file copy
                 shutil.copy2(absolute_filepath, output_filepath)
            # else:
                 # print(f"[{mp.current_process().name}] No changes needed for {relative_filepath}. Output file already exists.")


        doc.close()
        return (relative_filepath, 'success') # Indicate success

    except FileNotFoundError:
        print(f"[{mp.current_process().name}] Error: File not found at {absolute_filepath}")
        return (relative_filepath, 'failure')
    except pymupdf.FileDataError as e:
        print(f"[{mp.current_process().name}] Error processing file {relative_filepath}: {e} (Invalid file format or corrupted)")
        # Copy the original file to the errors directory
        os.makedirs(errors_subdir, exist_ok=True)
        shutil.copy2(absolute_filepath, errors_filepath)
        print(f"[{mp.current_process().name}] Copied original file to {errors_filepath}")
        return (relative_filepath, 'failure')
    except Exception as e:
        print(f"[{mp.current_process().name}] An unexpected error occurred while processing {relative_filepath}: {e}")
        # traceback.print_exc() # Uncomment for detailed error in worker logs
        # Copy the original file to the errors directory
        os.makedirs(errors_subdir, exist_ok=True)
        shutil.copy2(absolute_filepath, errors_filepath)
        print(f"[{mp.current_process().name}] Copied original file to {errors_filepath}")
        return (relative_filepath, 'failure')

# --- Main Script Logic ---
def get_pdf_list(input_path: pathlib.Path) -> list[tuple[str, str]]:
    """
    Scans the input path for PDF files and returns a list of (absolute_path, relative_path).
    Preserves directory structure.
    """
    pdf_files = []
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            pdf_files.append((str(input_path.resolve()), str(input_path.name)))
    elif input_path.is_dir():
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    absolute_path = os.path.join(root, file)
                    # Calculate relative path from the input_path root
                    relative_path = os.path.relpath(absolute_path, input_path)
                    pdf_files.append((absolute_path, relative_path))
    return pdf_files

def get_processed_list(output_root_dir: pathlib.Path, progress_file_name: str) -> set[str]:
    """
    Reads the progress file and returns a set of relative file paths already processed.
    """
    progress_filepath = output_root_dir / progress_file_name
    processed_files = set()
    if progress_filepath.exists():
        try:
            with open(progress_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_files.add(line.strip())
        except Exception as e:
            print(f"Warning: Could not read progress file {progress_filepath}: {e}")
    return processed_files

def record_progress(output_root_dir: pathlib.Path, progress_file_name: str, relative_filepath: str):
    """
    Appends the relative_filepath of a successfully processed file to the progress file.
    """
    progress_filepath = output_root_dir / progress_file_name
    try:
        with open(progress_filepath, 'a', encoding='utf-8') as f:
            f.write(relative_filepath + '\n')
    except Exception as e:
        print(f"Error: Could not write to progress file {progress_filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Find, process, and replace images in PDF files to JPEG, compress, and save, with parallel processing and resume."
    )
    parser.add_argument(
        "input_path",
        help="Path to the input PDF file or a directory containing PDF files."
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save the modified PDF files. Defaults to 'output_processed_pdfs' in the current directory.",
        default="output_processed_pdfs"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes to use. Defaults to 1 (no parallelization). Use -1 for number of CPU cores."
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        choices=range(0, 101),
        metavar="[0-100]",
        help="JPEG quality for the output images (0-100). Default is 95."
    )
    parser.add_argument(
        "--subset-fonts",
        action="store_true",
        help="Enable font subsetting for PDF files. Can significantly reduce file size for documents with large fonts."
    )
    parser.add_argument(
        "--errors-dir",
        help="Name of the subdirectory within the output directory for failed files. Defaults to 'errors_pdf'.",
        default="errors_pdf"
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=96, # Or 72 for more aggressive downsampling
        help="Target DPI for image downsampling. Images will be resized to fit this resolution based on their page bounding box. Default is 96."
    )
    parser.add_argument(
        "--force-color",
        choices=['rgb', 'gray', 'none'],
        default='none',
        help="Force image color conversion. 'rgb' to sRGB, 'gray' to grayscale, 'none' to keep original (default)."
    )
    # Argument to select text detection method
    parser.add_argument(
        "--text-detection-method",
        choices=['tesseract', 'opencv', 'none'],
        default='none',
        help="Method to use for text detection to adjust image resizing for readability. 'tesseract' requires Tesseract OCR (library and executable). 'opencv' requires OpenCV (library) and the hardcoded model file ('DB_TD500_resnet50.onnx' by default). 'none' disables text-based resizing. Default is 'none'."
    )
    parser.add_argument(
        "--min-text-height",
        type=int,
        default=10, # Minimum desired height for smallest text in pixels after resizing
        help="Minimum desired height (in pixels) for the smallest text found in images after resizing. Used when --text-detection-method is 'tesseract' or 'opencv'. Adjusts target DPI-based size upwards if needed for readability. Default is 10px."
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Attempt to use CUDA for OpenCV text detection if available. Requires OpenCV built with CUDA support."
    )

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    output_root_dir = pathlib.Path(args.output_dir)
    num_processes = args.jobs
    jpg_quality = args.quality
    enable_font_subsetting = args.subset_fonts
    errors_dir_name = args.errors_dir
    target_dpi = args.target_dpi
    force_color = args.force_color
    text_detection_method = args.text_detection_method
    min_readable_text_height_px = args.min_text_height
    use_cuda = args.use_cuda # Get the new argument
    progress_file_name = "processed_files.txt"

    # --- Hardcoded OpenCV Model Path (defined once in main) ---
    opencv_model_path = "DB_TD500_resnet50.onnx"
    # --- End Hardcoded Path ---


    # --- Validation for text detection methods ---
    if text_detection_method == 'tesseract':
        if not pytesseract_available:
            print("Error: Pytesseract library not found. Cannot use '--text-detection-method tesseract'.")
            print("Please install pytesseract (pip install pytesseract) or choose a different method.")
            sys.exit(1)
        try:
            # Check if Tesseract executable is found (pytesseract library might be installed, but not the executable)
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract executable not found. Cannot use '--text-detection-method tesseract'.")
            print("Please install Tesseract OCR and ensure it's in your system's PATH, or choose a different method.")
            sys.exit(1)
        except Exception as e:
             print(f"Warning: Could not fully verify Tesseract installation: {e}. Proceeding assuming it might work.")


    elif text_detection_method == 'opencv':
        if not opencv_available:
            print("Error: OpenCV (cv2) not found. Cannot use '--text-detection-method opencv'.")
            print("Please install opencv-python or opencv-contrib-python.")
            sys.exit(1)
        # Check for the model file existence upfront
        if not os.path.exists(opencv_model_path):
             print(f"Error: OpenCV model file not found at '{opencv_model_path}'.")
             print("Please ensure the model file exists at this path or update the 'opencv_model_path' variable in the script.")
             sys.exit(1)
        # Check for CUDA availability if requested
        if use_cuda:
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                print("Error: '--use-cuda' specified, but no CUDA-enabled GPU found or OpenCV was not built with CUDA support.")
                print("Please ensure you have a CUDA-enabled GPU and installed opencv-python with CUDA support (e.g., by building from source or using a specific wheel).")
                sys.exit(1)
            else:
                print(f"CUDA enabled GPU(s) found: {cv2.cuda.getCudaEnabledDeviceCount()}. Attempting to use CUDA for OpenCV DNN.")


    # --- End Validation ---


    if not input_path.exists():
        print(f"Error: Input path '{input_path}' not found.")
        sys.exit(1)

    # Ensure output root directory exists for the progress file and error directory
    output_root_dir.mkdir(parents=True, exist_ok=True)
    (output_root_dir / errors_dir_name).mkdir(parents=True, exist_ok=True)


    # Get list of all PDF files to potentially process, preserving structure
    all_pdf_files = get_pdf_list(input_path)
    if not all_pdf_files:
        print(f"No PDF files found in '{input_path}'.")
        sys.exit(0)

    # Get list of files already processed from the progress file
    processed_files = get_processed_list(output_root_dir, progress_file_name)

    # Filter out files that have already been processed
    files_to_process = [(abs_path, rel_path) for abs_path, rel_path in all_pdf_files if rel_path not in processed_files]

    if not files_to_process:
        print("All found PDF files have already been processed according to the progress file.")
        sys.exit(0)

    print(f"Found {len(all_pdf_files)} PDF files. {len(processed_files)} already processed.")
    print(f"Processing {len(files_to_process)} files...")

    # Prepare arguments for the worker function
    worker_args = [(abs_path, rel_path, str(output_root_dir.resolve()), errors_dir_name, jpg_quality, enable_font_subsetting, target_dpi, force_color, text_detection_method, min_readable_text_height_px)
                   for abs_path, rel_path in files_to_process]

    # Determine number of processes
    if num_processes == -1:
        num_processes = mp.cpu_count()
    elif num_processes <= 0:
         num_processes = 1 # Default to 1 process if non-positive value provided

    print(f"Using {num_processes} parallel processes.")

    # Use a multiprocessing Pool to distribute tasks
    # Use initializer to load the OpenCV model once per worker process
    # Pass the model path and the use_cuda flag to the initializer via initargs
    pool_initializer = None
    pool_initargs = ()
    if text_detection_method == 'opencv':
        # Only set initializer if OpenCV method is requested
        pool_initializer = worker_init
        pool_initargs = (opencv_model_path, use_cuda) # Pass model path AND use_cuda flag


    with mp.Pool(processes=num_processes, initializer=pool_initializer, initargs=pool_initargs) as pool:
        results = pool.imap_unordered(process_single_pdf, worker_args)

        for result_tuple in results:
            rel_path, status = result_tuple
            if status == 'success':
                record_progress(output_root_dir, progress_file_name, rel_path)
            elif status == 'failure':
                 # Worker already copied the file and printed the message
                 pass # Do nothing in the main loop for failed files
            # elif status == 'skipped': # Handle skipped files if implemented
            #      pass


    print("\nAll scheduled files processed.")

if __name__ == "__main__":
    # On Windows, wrap the main execution in if __name__ == '__main__':
    # This is necessary for multiprocessing to work correctly.
    # Also, set start method for consistency across platforms
    if sys.platform.startswith('win'):
        mp.freeze_support() # Needed for executables
        # 'spawn' is safer but might be slower than 'fork' (default on Unix)
        # 'fork' is not available on Windows
        # Setting it explicitly can help avoid issues
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # This can happen if set_start_method is called multiple times
            # in interactive environments or complex scripts.
            # In a simple script like this, it's less likely, but good practice
            # to handle if force=True isn't sufficient.
            pass


    main()