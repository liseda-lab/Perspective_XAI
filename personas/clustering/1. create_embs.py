import pandas as pd
import json
from sentence_transformers import SentenceTransformer

def generate_preference_embeddings(csv_file_path, output_file_path=None):
    """
    Generate 768-dimensional sentence embeddings for Preference column using all-mpnet-base-v2
    One embedding per user (combining all their preference sentences)
    
    Args:
        csv_file_path (str): Path to the CSV file
        output_file_path (str, optional): Path to save JSON output
    
    Returns:
        list: List of dictionaries with id, preference, and embedding
    """
    
    # Load the CSV file (using semicolon as delimiter based on file structure)
    df = pd.read_csv(csv_file_path, encoding='utf-8', delimiter=';')
    
    # Clean up any extra empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(subset=['ID', 'Preference'])
    
    # Load the Sentence-BERT model (all-mpnet-base-v2 produces 768-dim embeddings)
    print("Loading all-mpnet-base-v2 model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Extract preferences and generate embeddings
    preferences = df['Preference'].tolist()
    ids = df['ID'].tolist()
    
    print(f"Generating embeddings for {len(preferences)} users...")
    embeddings = model.encode(preferences, show_progress_bar=True)
    
    # Create result objects
    results = []
    for user_id, preference, embedding in zip(ids, preferences, embeddings):
        result_obj = {
            "id": user_id,
            "preference": preference,
            "embedding": embedding.tolist()  # Convert numpy array to list
        }
        results.append(result_obj)
    
    # Save to file if output path provided
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Processed {len(results)} users")
    print(f"- Embedding dimension: {len(results[0]['embedding'])} (should be 768)")
    print(f"- Model used: all-mpnet-base-v2")
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate one embedding per user
    results = generate_preference_embeddings('User_all_info.csv', 'user_preference_embeddings.json')
    
    # Display first result as example
    print(f"\nExample result for {results[0]['id']}:")
    print(f"Preference (first 150 chars): {results[0]['preference'][:150]}...")
    print(f"Embedding dimension: {len(results[0]['embedding'])}")
    print(f"First 5 embedding values: {results[0]['embedding'][:5]}")

# Alternative: Generate embeddings without saving to file
# results = generate_preference_embeddings('User_all_info.csv')

# Alternative: Access individual user embeddings
# for result in results:
#     user_id = result['id']
#     preference = result['preference'] 
#     embedding = result['embedding']  # 768-dimensional list
#     print(f"User {user_id} embedding dimension: {len(embedding)}")


    #CODE TO SEPARATE EMBEDDINGS PER COHORT
    import json
    with open('user_preference_embeddings.json') as f:
        user_embeddings = json.load(f)


    drec = ['user_01','user_02','user_03','user_04','user_05','user_06','user_07','user_08',
        'user_09','user_10','user_11','user_12','user_13','user_14','user_15','user_16',
        'user_17','user_18','user_19','user_20','user_21','user_22']

    dr = ['user_23','user_24','user_25','user_26','user_27','user_28','user_29','user_30',
        'user_31']

    dti = ['user_32','user_33','user_34','user_35','user_36','user_37']

    # save embeddings from each group
    drec_embs = [v for v in user_embeddings if v['id'] in drec]
    dr_embs = [v for v in user_embeddings if v['id'] in dr]
    dti_embs = [v for v in user_embeddings if v['id'] in dti]


     #Save the separated embeddings to JSON files
    with open('drec_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(drec_embs, f, ensure_ascii=False, indent=2)

    dr_dti_embs = dr_embs + dti_embs

    with open('dr_dti_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(dr_dti_embs, f, ensure_ascii=False, indent=2)