import os
import sys
import urllib.request
from tqdm import tqdm

RAW_URL = "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/resolve/main/data/train-00000-of-00001.parquet"
OUTPUT_PATH = os.path.join("data", "raw", "twitter_support_conversations.parquet")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_raw():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 100_000_000:
        print(f"File already exists: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)")
        return OUTPUT_PATH
    
    print(f"Downloading from {RAW_URL} to {OUTPUT_PATH}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(RAW_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(OUTPUT_PATH, 'wb') as out_file:
        total_size = int(resp.headers.get('Content-Length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading dataset") as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
    print(f"Download complete: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)")
    return OUTPUT_PATH

if __name__ == "__main__":
    download_raw()
