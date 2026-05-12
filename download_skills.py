import urllib.request
import zipfile
import os
import shutil
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://github.com/sickn33/antigravity-awesome-skills/archive/refs/heads/main.zip"
zip_path = "skills.zip"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, zip_path)
print("Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".agent_temp")

target_dir = ".agent/skills"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(os.path.dirname(target_dir), exist_ok=True)

# Move contents from extracted folder to destination
source_dir = ".agent_temp/antigravity-awesome-skills-main"
shutil.move(source_dir, target_dir)

# Cleanup
shutil.rmtree(".agent_temp")
os.remove(zip_path)
print(f"Skills successfully downloaded to {os.path.abspath(target_dir)}!")
