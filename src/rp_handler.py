import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.error
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = int(os.environ.get("COMFY_POLLING_INTERVAL_MS", 250))
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = int(os.environ.get("COMFY_POLLING_MAX_RETRIES", 500))
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# Logged once per worker process after Comfy responds; shows up in RunPod logs / job output.
_NODE_DIAGNOSTICS_LOGGED = False

# Nodes required by the character-sheet workflow (MVAdapter + Impact Pack + core).
_CHARACTER_SHEET_NODE_TYPES = (
    "LdmPipelineLoader",
    "DiffusersMVSchedulerLoader",
    "DiffusersMVModelMakeup",
    "DiffusersMVSampler",
    "FaceDetailer",
    "UltralyticsDetectorProvider",
)


def log_comfy_node_registry_once():
    """
    Query ComfyUI /object_info once and print whether critical custom nodes registered.
    If MVAdapter fails to import, these types are absent even when the directory exists on disk.
    """
    global _NODE_DIAGNOSTICS_LOGGED
    if _NODE_DIAGNOSTICS_LOGGED:
        return
    _NODE_DIAGNOSTICS_LOGGED = True
    url = f"http://{COMFY_HOST}/object_info"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            print(f"runpod-worker-comfy - object_info: unexpected JSON type {type(data)}")
            return
        print(
            f"runpod-worker-comfy - object_info: {len(data)} node type(s) registered"
        )
        for name in _CHARACTER_SHEET_NODE_TYPES:
            status = "PRESENT" if name in data else "MISSING"
            print(f"runpod-worker-comfy - object_info node {name!r}: {status}")
        missing = [n for n in _CHARACTER_SHEET_NODE_TYPES if n not in data]
        if missing:
            print(
                "runpod-worker-comfy - HINT: Check Comfy stderr above. Common fixes: (1) cv2/libgthread "
                "error → image needs apt package libglib2.0-0. (2) SCHEDULER_HANDLERS → Impact Pack "
                "must be tag 8.9 with ComfyUI 0.2.7, not Main. (3) UltralyticsDetectorProvider → "
                "ComfyUI-Impact-Subpack. (4) App workflow must use MVAdapter v1.0.2 API names "
                "DiffusersMVSchedulerLoader / DiffusersMVModelMakeup (not DiffusersSchedulerLoader). "
                "(5) numpy<2 for Impact 8.x after MVAdapter pip."
            )
    except requests.RequestException as e:
        print(f"runpod-worker-comfy - object_info diagnostic failed: {e}")


def _format_comfy_prompt_error(body: dict) -> str:
    """Flatten ComfyUI /prompt error JSON for logs and RunPod output."""
    err = body.get("error")
    if isinstance(err, dict):
        parts = [
            err.get("message"),
            err.get("type"),
            err.get("details"),
        ]
        text = "; ".join(str(p) for p in parts if p)
        return (text or json.dumps(err))[:2000]
    if err is not None:
        return str(err)[:2000]
    node_errors = body.get("node_errors")
    if node_errors:
        return json.dumps(node_errors)[:2000]
    return json.dumps(body)[:2000]


def _format_execution_status_error(status_obj: dict) -> str:
    """Flatten ComfyUI history status when status_str is error."""
    msgs = status_obj.get("messages")
    if isinstance(msgs, list) and msgs:
        try:
            return json.dumps(msgs)[:2000]
        except (TypeError, ValueError):
            return str(msgs)[:2000]
    return json.dumps(status_obj)[:2000]


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    # Return validated data and no error
    return {"workflow": workflow, "images": images}, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    for i in range(retries):
        try:
            response = requests.get(url)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")
                log_comfy_node_registry_once()
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
        server_address (str): The address of the ComfyUI server.

    Returns:
        list: A list of responses from the server for each image upload.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"runpod-worker-comfy - image(s) upload")

    for image in images:
        name = image["name"]
        image_data = image["image"]
        blob = base64.b64decode(image_data)

        # Prepare the form data
        files = {
            "image": (name, BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }

        # POST request to upload the image
        response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        if response.status_code != 200:
            upload_errors.append(f"Error uploading {name}: {response.text}")
        else:
            responses.append(f"Successfully uploaded {name}")

    if upload_errors:
        print(f"runpod-worker-comfy - image(s) upload with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"runpod-worker-comfy - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def queue_workflow(workflow):
    """
    Queue a workflow to be processed by ComfyUI.

    Returns:
        dict: Either ``{"prompt_id": "..."}`` on success or ``{"error": "..."}`` on validation/API failure.
    """
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{COMFY_HOST}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        raw = e.read() if e.fp else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "error": f"ComfyUI /prompt failed (HTTP {e.code}): {raw.decode(errors='replace')[:800]}"
            }
        if isinstance(body, dict):
            return {"error": _format_comfy_prompt_error(body)}
        return {"error": raw.decode(errors="replace")[:800]}

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "error": f"ComfyUI /prompt returned non-JSON: {raw.decode(errors='replace')[:500]}"
        }

    if not isinstance(body, dict):
        return {"error": f"ComfyUI /prompt unexpected response: {str(body)[:500]}"}

    if body.get("error") is not None:
        return {"error": _format_comfy_prompt_error(body)}

    prompt_id = body.get("prompt_id")
    if not prompt_id:
        return {
            "error": f"ComfyUI /prompt missing prompt_id; response: {json.dumps(body)[:800]}"
        }

    return {"prompt_id": prompt_id}


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())


def base64_encode(img_path):
    """
    Returns base64 encoded image.

    Args:
        img_path (str): The path to the image

    Returns:
        str: The base64 encoded image
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"{encoded_string}"


def process_output_images(outputs, job_id):
    """
    This function takes the "outputs" from image generation and the job ID,
    then determines the correct way to return the image, either as a direct URL
    to an AWS S3 bucket or as a base64 encoded string, depending on the
    environment configuration.

    Args:
        outputs (dict): A dictionary containing the outputs from image generation,
                        typically includes node IDs and their respective output data.
        job_id (str): The unique identifier for the job.

    Returns:
        dict: A dictionary with the status ('success' or 'error') and the message,
              which is either the URL to the image in the AWS S3 bucket or a base64
              encoded string of the image. In case of error, the message details the issue.

    The function works as follows:
    - It first determines the output path for the images from an environment variable,
      defaulting to "/comfyui/output" if not set.
    - It then iterates through the outputs to find the filenames of the generated images.
    - After confirming the existence of the image in the output folder, it checks if the
      AWS S3 bucket is configured via the BUCKET_ENDPOINT_URL environment variable.
    - If AWS S3 is configured, it uploads the image to the bucket and returns the URL.
    - If AWS S3 is not configured, it encodes the image in base64 and returns the string.
    - If the image file does not exist in the output folder, it returns an error status
      with a message indicating the missing image file.
    """

    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")

    output_images = []

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image in node_output["images"]:
                output_images.append(os.path.join(image["subfolder"], image["filename"]))

    print(f"runpod-worker-comfy - image generation is done")

    if not output_images:
        print("runpod-worker-comfy - no output images found in workflow results")
        return {
            "status": "error",
            "message": "No output images found in workflow results",
        }

    result_images = []
    use_s3 = os.environ.get("BUCKET_ENDPOINT_URL", False)

    for rel_path in output_images:
        local_image_path = os.path.join(COMFY_OUTPUT_PATH, rel_path)
        print("runpod-worker-comfy - processing generated image")

        if not os.path.exists(local_image_path):
            print("runpod-worker-comfy - WARNING: generated image file is missing")
            continue

        if use_s3:
            image = rp_upload.upload_image(job_id, local_image_path)
            print("runpod-worker-comfy - image uploaded to AWS S3")
        else:
            image = base64_encode(local_image_path)
            print("runpod-worker-comfy - image converted to base64")

        result_images.append(image)

    if not result_images:
        return {
            "status": "error",
            "message": "None of the output images exist",
        }

    return {
        "status": "success",
        "message": result_images if len(result_images) > 1 else result_images[0],
    }


def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    images = validated_data.get("images")

    # Make sure that the ComfyUI API is available
    if not check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {"error": "ComfyUI API is not reachable"}

    # Upload images if they exist
    upload_result = upload_images(images)

    if upload_result["status"] == "error":
        return upload_result

    # Queue the workflow
    try:
        queued_workflow = queue_workflow(workflow)
        if "error" in queued_workflow:
            err = queued_workflow["error"]
            print(f"runpod-worker-comfy - queue_workflow error: {err}")
            return {"error": err}
        prompt_id = queued_workflow["prompt_id"]
        print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}

    # Poll for completion
    print(f"runpod-worker-comfy - wait until image generation is complete")
    retries = 0
    try:
        while retries < COMFY_POLLING_MAX_RETRIES:
            history = get_history(prompt_id)

            if isinstance(history, dict) and prompt_id in history:
                entry = history[prompt_id]
                status_obj = entry.get("status")
                if isinstance(status_obj, dict) and status_obj.get("status_str") == "error":
                    detail = _format_execution_status_error(status_obj)
                    print(f"runpod-worker-comfy - ComfyUI execution error: {detail}")
                    return {"error": f"ComfyUI workflow failed: {detail}"}
                if entry.get("outputs"):
                    break

            time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
            retries += 1
        else:
            return {"error": "Max retries reached while waiting for image generation"}
    except Exception as e:
        return {"error": f"Error waiting for image generation: {str(e)}"}

    # Get the generated image and return it as URL in an AWS bucket or as base64
    images_result = process_output_images(history[prompt_id].get("outputs"), job["id"])

    result = {**images_result, "refresh_worker": REFRESH_WORKER}

    return result


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
