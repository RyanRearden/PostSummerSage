from sentence_transformers import SentenceTransformer, util 

model = SentenceTransformer('all-MiniLM-L6-v2')

# Example descriptions
ai_description = """The image is an aerial view of a busy street at night. The street is lined with buildings on both sides and there are several cars parked on the right side of the road. The buildings are lit up with colorful lights, creating a vibrant atmosphere. In the center of the image, there is a zebra crossing with white lines and a red crosswalk. There are several people walking on the sidewalk, some of them are carrying bags, while others are walking towards the intersection. The sky is dark, indicating that it is nighttime. LABELS: several cars, The buildings, a zebra crossing, several people, bags, others, The sky, police car at night with flashing lights, black car on street at night, police car on city street at dusk, police car, CCTV footage of person walking on street with black bag, CCTV image of man in black hoodie walking on sidewalk, CCTV camera footage of man walking on stairs at night, CCTV images of people walking on escalator at night, CCTV video of man with backpack and camera, CCTV photo of man on escalators at night, CCTV of man wearing black hooded jacket and white shirt, CCTV scan of man's face with black jacket and blue jeans, CCTV view of man and woman walking on busy street at street at Night, CCTV surveillance image of police car at traffic light, CCTV screen of man suspected of murder at night, CCCCTV screenshot of man carrying woman on shoulder at night, CCTV still of woman with long black hair and blue jacket, CCTV security camera footage, CCTVTV footage, CCTV scene of man holding gun at night, traffic light"""

ground_truth_description = """The image is of a street intersection at night. The street is lined with buildings on the right side and there are several cars driving on the road. In the foreground of the image, there are two zebra crossings; one is going horizontally while the other is going vertically. There are people on the zebra crossing and sidewalk. It is night. """

# Compute embeddings
ai_embedding = model.encode([ai_description])
gt_embedding = model.encode([ground_truth_description])

# Compute cosine similarity
similarity_score = util.cos_sim(ai_embedding, gt_embedding).item()
print(f"Semantic Similarity Score: {similarity_score:.4f}")


import nltk 
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Tokenize and preprocess
def preprocess(text):
    return set(word_tokenize(text.lower()))

ai_tokens = preprocess(ai_description)
ground_truth_tokens = preprocess(ground_truth_description)

# Calculate precision, recall, F1
true_positives = len(ai_tokens & ground_truth_tokens)
precision = true_positives / len(ai_tokens)
recall = true_positives / len(ground_truth_tokens)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
