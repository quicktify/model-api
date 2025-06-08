import json
import mimetypes
import tempfile

from google.cloud import storage


def upload_json_to_gcs(data: dict, bucket_name: str, destination_blob_name: str) -> str:
    """
    Uploads a JSON-serializable dict to Google Cloud Storage and returns the public URL.
    """
    # Simpan data ke file sementara
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp)
        tmp.flush()
        tmp_path = tmp.name

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(tmp_path, content_type="application/json")

    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"


def upload_image_to_gcs(
    image_path: str, bucket_name: str, destination_blob_name: str
) -> str:
    """
    Uploads an image file to Google Cloud Storage and returns the public URL.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    content_type, _ = mimetypes.guess_type(image_path)
    blob.upload_from_filename(image_path, content_type=content_type or "image/png")
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
