import sage_data_client
import pandas as pd
import requests
import os

USERNAME = ""
USERTOKEN = ''



# Output directory for images
output_image_dir = "downloaded_images_W085"
os.makedirs(output_image_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to download an image using an authenticated session
def download_image_with_session(session, image_url, output_dir, filename):
    try:
        response = session.get(image_url, stream=True)
        response.raise_for_status()
        image_path = os.path.join(output_dir, filename)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path  # Return the local path to the downloaded image
    except requests.RequestException as e:
        print(f"Failed to download image: {image_url}, error: {e}")
        return None

# Query the data
df = sage_data_client.query(
    start="2024-8-01T07:00:00.000Z",
    end="2024-11-22T08:00:00.000Z", 
    filter={
        "plugin": "registry.sagecontinuum.org/yonghokim/plugin-image-captioning:0.1.0.*",
        "vsn": "W085"
    }
)

# Filter data for the two relevant "name" values
description_df = df[df['name'] == "env.image.description"]
upload_df = df[df['name'] == "upload"]

# Merge the two DataFrames based on the "timestamp" column
paired_df = pd.merge(description_df, upload_df, on="timestamp", suffixes=('_description', '_upload'))

# Randomly select 25 pairs
if len(paired_df) >= 25:  # Ensure there are at least 25 pairs
    random_pairs = paired_df.sample(n=25, random_state=42)  # Random state for reproducibility
else:
    print("Not enough paired data to sample 25 rows.")
    random_pairs = paired_df  # Use all available data

# Prepare data for CSV output
if not random_pairs.empty:  # Ensure there are random pairs
    output_data = []
    
    with requests.Session() as session:
        session.auth = (USERNAME, USERTOKEN)  # Authenticate the session

        for _, row in random_pairs.iterrows():
            # Construct the image URL (if necessary)
            image_url = row['value_upload']  # Assuming 'url_upload' already contains the full URL
            description = row['value_description']  # Assuming the description value is in 'value_description'

            # Download the image and save locally
            image_filename = f"{row['timestamp']}.jpg"  # Use timestamp for unique filenames
            image_path = download_image_with_session(session, image_url, output_image_dir, image_filename)
            
            if image_path:
                # Append to the output data
                output_data.append({
                    "Image Path": image_path,
                    "Description": description
                })
    
    # Create a DataFrame for the output
    output_df = pd.DataFrame(output_data)

    # Save to CSV
    output_df.to_csv("image_description_pairs_W085.csv", index=False)
    print("CSV file created: image_description_pairs_W085.csv")
else:
    print("No random pairs to store.")