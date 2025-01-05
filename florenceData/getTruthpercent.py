import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize
import re
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

nltk.download("punkt_tab")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing function for tokenization and cleaning
def preprocess(text):
    text = re.sub(r"(DESCRIPTION:  |LABELS:)", "", str(text), flags=re.IGNORECASE).strip()
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

# Function to calculate BLEU score
def calculate_bleu(ai_description, ground_truth):
    return corpus_bleu([ai_description], [[ground_truth]]).score

# Function to calculate ROUGE scores
def calculate_rouge(ai_description, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, ai_description)
    return scores

# Process the CSV file and calculate metrics
def process_csv(file_path):
    df = pd.read_csv(file_path)
    
    total_similarity = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_bleu = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    count = 0
    
    for _, row in df.iterrows():
        ai_description = row['Description']
        ground_truth = row['Ground Truth']
        
        # Preprocessed descriptions               
        #ai_description = preprocess(ai_description)
        #ground_truth = preprocess(ground_truth)
        
        similarity_score = calculate_similarity(ai_description, ground_truth)
        precision, recall, f1_score = calculate_metrics(ai_description, ground_truth)
        bleu_score = calculate_bleu(ai_description, ground_truth)
        rouge_scores = calculate_rouge(ai_description, ground_truth)
        
        print(f"AI Description: {ai_description}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
        print("=====")
        
        # Accumulate results
        total_similarity += similarity_score
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score
        total_bleu += bleu_score
        total_rouge1 += rouge_scores['rouge1'].fmeasure
        total_rouge2 += rouge_scores['rouge2'].fmeasure
        total_rougeL += rouge_scores['rougeL'].fmeasure
        count += 1
    
    # Calculate averages
    avg_similarity = total_similarity / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_f1 = total_f1 / count
    avg_bleu = total_bleu / count
    avg_rouge1 = total_rouge1 / count
    avg_rouge2 = total_rouge2 / count
    avg_rougeL = total_rougeL / count
    
    print(f"\nFinal Averages:")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

file_path = "/home/ryanrearden/Documents/SAGE_fromLaptop/PostSummerSage/SAGE_Img_Data - W07B.csv"
process_csv(file_path)
