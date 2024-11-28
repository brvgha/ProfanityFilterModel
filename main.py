# imports
from setup import setup
setup()
import logging
import os
from dotenv import load_dotenv
import PIL
import sklearn.model_selection as model_selection
import requests as r
import numpy as np
import pandas as pd
import praw as p
import keras as k
import tensorflow as tf

logger = logging.Logger('./profanityFilterModel.log')

load_dotenv()
# functions
# function to preprocess images for learning


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


def decode_image(image, image_format):
    image_format = tf.strings.lower(image_format)

    if tf.equal(image_format, b'jpeg') or tf.equal(image_format, b'jpg'):
        return tf.cast(tf.image.decode_jpeg(image, channels=3), dtype=tf.float32)
    elif tf.equal(image_format, b'png'):
        return tf.cast(tf.image.decode_png(image, channels=3), dtype=tf.float32)
    elif tf.equal(image_format, b'gif'):
        return tf.cast(tf.image.decode_gif(image), dtype=tf.float32)
    elif tf.equal(image_format, b'bmp'):
        return tf.cast(tf.image.decode_bmp(image, channels=3), dtype=tf.float32)
    else:
        # Return a zero tensor with the same dtype and shape as expected
        return tf.zeros([128, 128, 3], dtype=tf.float32)

# Function to load and preprocess images


def preprocess_image(image_path, label, img_size=(128, 128)):
    # Load the image
    image = tf.io.read_file(image_path)
    image_format = tf.strings.lower(tf.strings.split(image_path, ".")[-1])
    try:
        image = tf.cond(
            tf.reduce_any(
                tf.equal(image_format, [b'jpeg', b'jpg', b'png', b'gif', b'bmp'])),
            lambda: decode_image(image, image_format),
            lambda: tf.zeros([img_size[0], img_size[1], 3],
                             dtype=tf.float32)  # Default zero tensor
        )
    except ValueError as e:
        print(e)
        return None, None

    image = tf.ensure_shape(image, [None, None, 3])
    image = tf.image.resize(image, img_size)        # Resize to desired size
    image = image / 255.0                           # Normalize to [0, 1]
    return image, label


def preprocess_new_image(image_path, img_size=(128, 128)):
    # Load the image
    image = tf.io.read_file(image_path)
    # Adjust if your images are not JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize to model's input size
    image = tf.image.resize(image, img_size)
    image = image / 255.0                           # Normalize to [0, 1]
    return tf.expand_dims(image, axis=0)            # Add batch dimension


logger.info("Functions loaded")

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


def fetch_posts(list_of_subreddits):
    list_of_urls = []
    for subreddit in list_of_subreddits:
        subreddit = reddit.subreddit(subreddit)
        for submission in subreddit.hot(limit=500):
            url = submission.url
            if url.endswith(('.jpg', '.png', '.jpeg')):
                list_of_urls.append(url)
    return list_of_urls


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

train_df, val_df = model_selection.train_test_split(df, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['images'].values, train_df['label'].values))
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df['images'].values, val_df['label'].values))

logger.info('Cleaned training and validation datasets ready')

# Preprocess and batch the data
batch_size = 32

train_dataset = (
    train_dataset
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

logger.info('Preprocess complete')

# Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    # Adjust for the number of classes
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Use categorical_crossentropy for multi-class
    metrics=['accuracy']
)

logger.info('Model created and compiled, beginning training...')

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

logger.info('Model trained.')

# sfw more links: https://th.bing.com/th/id/R.2b8e763cac892f3600a9c39e8c080798?rik=KE6r1yWXNeqmzw&pid=ImgRaw&r=0 https://keyassets.timeincuk.net/inspirewp/live/wp-content/uploads/sites/8/2021/02/GettyImages-997141470-e1614176377827-620x540.jpg
sfw_test = r.get(
    'https://th.bing.com/th/id/R.2b8e763cac892f3600a9c39e8c080798?rik=KE6r1yWXNeqmzw&pid=ImgRaw&r=0')
with open(save_path + "sfw_test_image" + ".jpeg", 'wb') as file:
    file.write(sfw_test.content)
    file.close()
# more links: https://static-ca-cdn.eporner.com/gallery/Zb/nS/4TOpbKLnSZb/7849956-ai-generated-porn-30215c4198b709c4_880x660.jpg https://th.bing.com/th/id/OIP.7LfDEWaf-nv4Y5gRka18HAHaLG?rs=1&pid=ImgDetMain https://th.bing.com/th/id/R.f19f057b4e68ff5022c02797d09c26ba?rik=iQy89tPMgl%2bQhg&pid=ImgRaw&r=0 https://static-ca-cdn.eporner.com/gallery/Hn/fn/biY9rU3fnHn/614061-angela-white-nude.jpg
nsfw_test = r.get(
    'https://static-ca-cdn.eporner.com/gallery/Hn/fn/biY9rU3fnHn/614061-angela-white-nude.jpg')
with open(save_path + "nsfw_test_image" + ".jpeg", 'wb') as file:
    file.write(nsfw_test.content)
    file.close()

logger.info('Beginning Predictions')
sfw_input_image_path = save_path + "sfw_test_image" + ".jpeg"
nsfw_input_image_path = save_path + "nsfw_test_image" + ".jpeg"
sfw_new_image = preprocess_new_image(sfw_input_image_path)
nsfw_new_image = preprocess_new_image(nsfw_input_image_path)
# sfw prediction
prediction_sfw = model.predict(sfw_new_image)[0][0]
predicted_label_sfw = label_mapping.get(
    'sfw' if prediction_sfw > 0.5 else 'nsfw')

predicted_label_sfw_string = 'nsfw' if prediction_sfw > 0.5 else 'sfw'
print(f"Predicted Probability: {prediction_sfw:.2f}")
print(f"Predicted Label: {predicted_label_sfw_string}")

# nsfw prediction
prediction_nsfw = model.predict(nsfw_new_image)[0][0]
predicted_label_nsfw = label_mapping.get(
    'sfw' if prediction_nsfw > 0.5 else 'nsfw')

predicted_label_nsfw_string = 'nsfw' if prediction_nsfw > 0.5 else 'sfw'
print(f"Predicted Probability: {prediction_nsfw:.2f}")
print(f"Predicted Label: {predicted_label_nsfw_string}")

logger.info('Process complete, saving model')

model.save('./profanity_filter_model')
