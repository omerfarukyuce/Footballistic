"""
Utility script to download sample football footage for testing.
"""

import os
import requests


def download_file_from_google_drive(file_id: str, destination: str):
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(url, params={'id': file_id}, stream=True)
    
    # Handle Google Drive confirmation token for larger files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
        
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    
    CHUNK_SIZE = 32768
    total_bytes = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                
    print(f"Downloaded {total_bytes / (1024*1024):.2f} MB to {destination}")


if __name__ == "__main__":
    # 08fd33_0.mp4 clip from DFL Bundesliga / Roboflow Football dataset
    sample_id = "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
    dest_path = os.path.join("data", "sample_videos", "sample.mp4")
    
    print(f"Downloading sample football match clip to {dest_path}...")
    download_file_from_google_drive(sample_id, dest_path)

