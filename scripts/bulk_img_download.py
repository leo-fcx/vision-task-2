import os
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS

def download_images(query, save_dir, max_images=100):
    os.makedirs(save_dir, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        for idx, result in enumerate(results):
            try:
                url = result["image"]
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                filename = os.path.join(save_dir, f"{query.replace(' ', '_')}_{idx:03}.jpg")
                img.save(filename)
                print(f"[{idx+1}] Saved: {filename}")
            except Exception as e:
                print(f"Failed to download image {idx+1}: {e}")

# Imagenes para entrenamiento
# download_images("iPhone 13 front", "data/subclasses/iphone", max_images=25)
# download_images("iPhone 13 back", "data/subclasses/iphone", max_images=25)
# download_images("iPhone 13 in hand", "data/subclasses/iphone", max_images=25)
# download_images("iPhone 13 on table", "data/subclasses/iphone", max_images=25)
# download_images("Samsung Galaxy S22 front", "data/subclasses/samsung", max_images=25)
# download_images("Samsung Galaxy S22 back", "data/subclasses/samsung", max_images=25)
# download_images("Samsung Galaxy S22 in hand", "data/subclasses/samsung", max_images=25)
# download_images("Samsung Galaxy S22 on table", "data/subclasses/samsung", max_images=25)
# download_images("Redmi Note 12 front", "data/subclasses/redmi", max_images=25)
# download_images("Redmi Note 12 back", "data/subclasses/redmi", max_images=25)
# download_images("Redmi Note 12 in hand", "data/subclasses/redmi", max_images=25)
# download_images("Redmi Note 12 on table", "data/subclasses/redmi", max_images=25)

# Imagenes para testeo
# download_images("iPhone 13 front", "data/raw_images", max_images=2)
# download_images("iPhone 13 back", "data/raw_images", max_images=2)
download_images("iPhone 13 in hand", "data/raw_images", max_images=2)
download_images("iPhone 13 on table", "data/raw_images", max_images=2)
# download_images("Samsung Galaxy S22 front", "data/raw_images", max_images=2)
# download_images("Samsung Galaxy S22 back", "data/raw_images", max_images=2)
download_images("Samsung Galaxy S22 in hand", "data/raw_images", max_images=2)
download_images("Samsung Galaxy S22 on table", "data/raw_images", max_images=2)
# download_images("Redmi Note 12 front", "data/raw_images", max_images=2)
# download_images("Redmi Note 12 back", "data/raw_images", max_images=2)
download_images("Redmi Note 12 in hand", "data/raw_images", max_images=2)
download_images("Redmi Note 12 on table", "data/raw_images", max_images=2)

# download_images("iPhone 13 angled view", "data/raw_images", max_images=2)
# download_images("iPhone 13 side view", "data/raw_images", max_images=2)
# download_images("iPhone 13 with case", "data/raw_images", max_images=2)
# download_images("iPhone 13 on desk", "data/raw_images", max_images=2)

# download_images("Samsung Galaxy S22 angled view", "data/raw_images", max_images=2)
# download_images("Samsung Galaxy S22 side view", "data/raw_images", max_images=2)
# download_images("Samsung Galaxy S22 with case", "data/raw_images", max_images=2)
# download_images("Samsung Galaxy S22 on desk", "data/raw_images", max_images=2)

# download_images("Redmi Note 12 angled view", "data/raw_images", max_images=2)
# download_images("Redmi Note 12 side view", "data/raw_images", max_images=2)
# download_images("Redmi Note 12 with case", "data/raw_images", max_images=2)
# download_images("Redmi Note 12 on desk", "data/raw_images", max_images=2)

