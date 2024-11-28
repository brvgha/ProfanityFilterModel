import praw as p
import logging
import os
import pandas as pd
import requests as r
import PIL
import tensorflow as tf

def validate_image_with_pil(image_path):
    try:
        with PIL.Image.open(image_path) as img:
            img.verify()  # Verify the file is a valid image
            img = PIL.Image.open(image_path)  # Reopen to check file integrity
            img.load()  # Load the image data
            if img.format not in ['JPEG', 'PNG', 'GIF', 'BMP']:
                raise ValueError(f"Invalid image format: {img.format}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {img.size}")
            if img.mode != 'RGB':
                raise ValueError(f"Invalid image mode: {img.mode}")
            return True
    except Exception as e:
        print(f"Invalid image: {image_path}, Error: {e}")
        return False

def validate_image_with_tf(image_path):
    try:
        # Attempt to read and decode the image using TensorFlow
        image = tf.io.read_file(image_path)
        decoded_image = tf.image.decode_image(image, channels=3)

        # Check if decoded image has valid dimensions
        if decoded_image.shape.rank != 3 or decoded_image.shape[2] != 3:
            return False
        return True
    except Exception as e:
        print(f"Invalid image: {image_path}, Error: {e}")
        return False
    
def fetch_posts(list_of_subreddits):
    list_of_urls = []
    for subreddit in list_of_subreddits:
        subreddit = reddit.subreddit(subreddit)
        for submission in subreddit.hot(limit=500):
            url = submission.url
            if url.endswith(('.jpg', '.png', '.jpeg')):
                list_of_urls.append(url)
    return list_of_urls

logger = logging.Logger(
    './redditimages.log'
)

logger.info("Starting getRedditImages.py")

reddit = p.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT'),
    username=os.getenv('REDDIT_USERNAME'),
    password=os.getenv('REDDIT_PASSWORD'),
    check_for_async=False
)

logger.info('Reddit initialised')

sfw_subreddits = ['awww', 'SFW', 'gaming', 'movies', 'books', 'Music', 'funny',
                  'pics', 'memes', 'science', 'history', 'food',
                  'television', 'politics', 'space', 'mildlyinteresting']
nsfw_subreddits = ['nsfw', 'gettingherselfoff', 'Nude_Selfie', 'bdsm', 'BBW',
                   'Boobies']
label_mapping = {'sfw': 0, 'nsfw': 1}





nsfw_return = fetch_posts(nsfw_subreddits)
sfw_return = fetch_posts(sfw_subreddits)
nsfw = {
    'url': nsfw_return,
    'label': ['nsfw' for x in range(len(nsfw_return))]
}

sfw = {
    'url': sfw_return,
    'label': ['sfw' for x in range(len(sfw_return))]
}

combined_list = nsfw | sfw

logger.info('Reddit post urls gotten')

# ready the df for training
df = pd.DataFrame({
    'url': nsfw['url'] + sfw['url'],
    'label': nsfw['label'] + sfw['label']
})
df['label'] = df['label'].map(label_mapping)
save_path = "./images/"
images = []
i = 0
for url in df['url']:
    try:
        response = r.get(url, timeout=5)
        if response.status_code == 200:
            image_path = save_path + "image" + str(i) + ".jpeg"
            with open(image_path, 'wb') as file:
                file.write(response.content)
                file.close()
            print(f"Image successfully downloaded: {image_path}")
            images.append(save_path + "image" + str(i) + ".jpeg")
            i += 1
        else:
            print(
                f"Failed to download image. Status code: {response.status_code}")
            images.append(None)
    except Exception as e:
        print(f"An error occurred: {e}")
        images.append(None)

logger.info('Images downloaded')

# this gets rid of the nas for the following TF steps
df['images'] = images
df = df.dropna()

df['valid_images'] = df['images'].apply(validate_image_with_pil)
df = df[df['valid_images'] == True]
df['valid_images'] = df['images'].apply(validate_image_with_tf)
df = df[df['valid_images'] == True]
df = df.drop(columns=['valid_images'])