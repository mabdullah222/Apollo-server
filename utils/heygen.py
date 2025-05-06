import requests
import time
import os
import re
from datetime import datetime

def generate_heygen_video(text, output_folder="lecVids", avatar_id="Daisy-inskirt-20220818", background="#008000"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if len(text) > 3000:
        print(f"Warning: Text length ({len(text)}) exceeds recommended limit. Truncating.")
        text = text[:2997] + "..."
    
    api_key = os.environ['HEYGEN_API_KEY']
    
    url = 'https://api.heygen.com/v2/video/generate'
    
    headers = {
        'X-Api-Key': api_key,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    background_config = {
        "type": "color",
        "value": background
    }
    if not background.startswith("#"):
        background_config = {
            "type": "image",
            "value": background
        }
    
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": text,
                    "voice_id": "2d5b0e6cf36f460aa7fc47e3eee4ba54"
                },
                "background": background_config
            }
        ],
        "dimension": {
            "width": 1280,
            "height": 720
        },
        "test": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response_data.get('error') is None:
            video_id = response_data['data']['video_id']
            print(f"Video is being processed. Video ID: {video_id}")
            
            status_url = f'https://api.heygen.com/v1/video_status.get?video_id={video_id}'
            max_retries = 30
            retry_count = 0
            
            while retry_count < max_retries:
                status_response = requests.get(status_url, headers=headers)
                status_data = status_response.json()
                
                if 'data' not in status_data or 'status' not in status_data['data']:
                    print(f"Unexpected response: {status_data}")
                    time.sleep(10)
                    retry_count += 1
                    continue
                    
                status = status_data['data']['status']
                
                if status == 'completed':
                    video_url = status_data['data']['video_url']
                    print(f"Video processing complete. Downloading from: {video_url}")
                    
                    video_response = requests.get(video_url)
                    if video_response.status_code == 200:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        safe_id = re.sub(r'[^\w\s-]', '', video_id[:8])
                        file_path = os.path.join(output_folder, f"lecture_{safe_id}_{timestamp}.mp4")
                        
                        with open(file_path, 'wb') as f:
                            f.write(video_response.content)
                        print(f"Video downloaded successfully to: {file_path}")
                        return file_path
                    else:
                        return f"Failed to download video: HTTP {video_response.status_code}"
                
                elif status == 'failed':
                    return "Video processing failed."
                
                else:
                    print("Video is still processing. Checking again in 10 seconds...")
                    time.sleep(10)
                    retry_count += 1
            
            return "Video processing timed out after maximum retries."
        else:
            return f"Error: {response_data['error']['message']}"
    
    except Exception as e:
        return f"An error occurred: {str(e)}"