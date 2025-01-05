import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing function for tokenization and cleaning
def preprocess(text):
    return set(word_tokenize(text.lower()))

# Function to calculate precision, recall, and F1 score
def calculate_metrics(ai_description, ground_truth):
    ai_tokens = preprocess(ai_description)
    gt_tokens = preprocess(ground_truth)
    
    # Calculate precision, recall, F1
    true_positives = len(ai_tokens & gt_tokens)
    precision = true_positives / len(ai_tokens) if ai_tokens else 0
    recall = true_positives / len(gt_tokens) if gt_tokens else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# Function to calculate cosine similarity
def calculate_similarity(ai_description, ground_truth):
    ai_embedding = model.encode([ai_description])
    gt_embedding = model.encode([ground_truth])
    similarity_score = util.cos_sim(ai_embedding, gt_embedding).item()
    return similarity_score

# Process the CSV file
def clean_ground_truth(ground_truth):
    # Convert to string to handle potential NaN values
    ground_truth = str(ground_truth)
    
    # Extract description (remove 'DESCRIPTION:' if present)
    description_match = re.search(r"(?i)DESCRIPTION:\s*(.*?)(?=\n?LABELS:|\Z)", ground_truth, re.DOTALL)
    description = description_match.group(1).strip() if description_match else ground_truth.strip()
    
    # Remove any redundant 'Description' text
    description = re.sub(r"(?i)^description\s*", "", description).strip()
    
    # Extract labels
    labels_match = re.findall(r"(?i)[-\*]?\s*([^-\n]+)", ground_truth)
    
    # Clean labels
    labels = [label.strip() for label in labels_match if label.strip()]
    
    # If description is empty and labels exist, use first label as description
    if not description and labels:
        description = labels.pop(0)
    
    # Combine description and labels in the desired format
    # Only add labels if they exist
    full_text = f"DESCRIPTION: {description}"
    if labels:
        full_text += f"  {', '.join(labels)}"
    
    return full_text

def preprocess_description(row):
    # Extract the description and labels
    description = row['Description']
    ground_truth = str(row['Ground Truth'])
    
    # Remove 'DESCRIPTION:' label if present
    cleaned_description = re.sub(r"(?i)\bDESCRIPTION:\s*", "", description).strip()
    
    # Remove any redundant 'Description' text
    cleaned_description = re.sub(r"(?i)^description\s*", "", cleaned_description).strip()
    
    # Extract labels from ground truth
    labels_match = re.search(r"(?i)LABELS:([\s\S]+)", ground_truth)
    labels = []
    if labels_match:
        # Clean up labels: remove dashes, newlines, extra spaces
        raw_labels = labels_match.group(1).strip()
        labels = [label.strip() for label in re.sub(r"[\-\n]", "", raw_labels).split(',')]
        # Remove empty labels
        labels = [label for label in labels if label]
    
    # If description is empty and labels exist, use first label as description
    if not cleaned_description and labels:
        cleaned_description = labels.pop(0)
    
    # Combine description and labels in the desired format
    formatted_output = f"DESCRIPTION: {cleaned_description}"
    if labels:
        formatted_output += f"  {', '.join(labels)}"
    
    return formatted_output

def process_csv(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Initialize metrics
    total_similarity = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0
    
    # Variables to track highest and lowest similarities
    highest_similarities = []
    lowest_similarities = []

    for _, row in df.iterrows():
        # Use the new preprocessing function
        formatted_description = preprocess_description(row)
        
        # Clean the ground truth
        cleaned_ground_truth = clean_ground_truth(row['Ground Truth'])
        
        # Calculate similarity and metrics
        similarity_score = calculate_similarity(formatted_description, cleaned_ground_truth)
        precision, recall, f1_score = calculate_metrics(formatted_description, cleaned_ground_truth)
        
        # Print formatted version and metrics
        print(f"AI Description: {formatted_description}")
        print(f"Ground Truth: {cleaned_ground_truth}")
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print("---")
        
        # Track the highest and lowest similarities
        highest_similarities.append((similarity_score, row['Image Path']))
        lowest_similarities.append((similarity_score, row['Image Path']))
        
        highest_similarities = sorted(highest_similarities, reverse=True)[:2]
        lowest_similarities = sorted(lowest_similarities)[:2]
        
        # Accumulate results
        total_similarity += similarity_score
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score
        count += 1
    
    # Calculate averages
    avg_similarity = total_similarity / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_f1 = total_f1 / count
    
    # Print final averages
    print(f"\nFinal Averages:")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Print top 2 highest and lowest similarities
    print("\nTop 2 Highest Similarity Scores:")
    for score, image_name in highest_similarities:
        print(f"Image: {image_name}, Similarity: {score:.4f}")
    
    print("\nTop 2 Lowest Similarity Scores:")
    for score, image_name in lowest_similarities:
        print(f"Image: {image_name}, Similarity: {score:.4f}")


# Run the program
file_path = ""  # Replace with your CSV file path
process_csv(file_path)
