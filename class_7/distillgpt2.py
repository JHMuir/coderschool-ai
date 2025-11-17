"""

Heirarchy of typical Model deployment: 
1. Training from scratch
    (requires massive data & compute)
2. Pre-training (GPT-2)
    (enables transfer learning)
3. Distillation (DistilGPT-2)
    (enables efficient deployment)
4. Fine-tuning
    (enables task-specific adaptation)
5. Your Model

Distillation - a student learns from a teacher by copying the teacher's outputs, not actually being trained
             - Normal training uses cross-entropy loss with hard targets
             - Distillation uses KL divergence (measuring difference between distributions)

Pre-Training - model learns general language understanding from massive datasets
Fine-Tuning - train the same model on a smaller, specific dataset to adapt it for a particular task

Transformers - a massive library for working with pre-trained language models and adapting them to fit our needs. Built on top of PyTorch. 
"""

import pandas
import transformers
import requests
import zipfile
import os
import re
import random
import torch

def download_cornell_corpus():
    """
    Download the Cornell Movie Dialogs Corpus if we don't have it yet.
    """
    if not os.path.exists('cornell_movie_dialogs'):        
        # Download the zip file
        print("Downloading cornell movie dialogue corpus dataset")
        response = requests.get("http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip")
        with open('cornell_corpus.zip', 'wb') as f:
            f.write(response.content)
        
        # Extract it
        with zipfile.ZipFile('cornell_corpus.zip', 'r') as zip_ref:
            zip_ref.extractall('cornell_movie_dialogs')
        
        print("Download complete!")
    else:
        print("Cornell corpus already downloaded.")

def load_movie_lines(corpus_path='cornell_movie_dialogs/cornell movie-dialogs corpus'):
    """
    Load all the movie lines from the corpus.
    
    The movie_lines.txt file has lines in this format:
    L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    """
    lines_file = os.path.join(corpus_path, 'movie_lines.txt')
    
    # Read the file line by line
    lines_data = []
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(lines_file, 'r', encoding=encoding) as f:
                for line in f:
                    # Split by the separator +++$+++
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) == 5:
                        lines_data.append({
                            'line_id': parts[0],
                            'character_id': parts[1],
                            'movie_id': parts[2],
                            'character_name': parts[3],
                            'text': parts[4]
                        })
            break
        except UnicodeDecodeError:
            continue
    
    return pandas.DataFrame(lines_data)

def load_movie_titles(corpus_path='cornell_movie_dialogs/cornell movie-dialogs corpus'):
    """
    Load movie titles and metadata.
    
    The movie_titles.txt file has lines in this format:
    m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ 62847 +++$+++ ['comedy', 'romance']
    
    We parse this into a DataFrame with: movie_id, title, year, rating, votes, genres
    """
    titles_file = os.path.join(corpus_path, 'movie_titles_metadata.txt')
    
    titles_data = []
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(titles_file, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) == 6:
                        titles_data.append({
                            'movie_id': parts[0],
                            'title': parts[1],
                            'year': parts[2],
                            'rating': parts[3],
                            'votes': parts[4],
                            'genres': parts[5]
                        })
            break
        except UnicodeDecodeError:
            continue
    
    return pandas.DataFrame(titles_data)

def find_movies_for_character(lines_df, titles_df, character_name):
    """
    Find all movies that a character appears in.
    
    This is useful because many character names (like JACK, JOHN, SARAH)
    appear in multiple movies. We want to let the user choose which specific
    version of the character they want.
    """
    # Get all lines for this character
    character_lines = lines_df[
        lines_df['character_name'].str.upper() == character_name.upper()
    ]
    
    if len(character_lines) == 0:
        print(f"No character named '{character_name}' found in corpus.")
        return None
    
    # Group by movie and count lines
    movie_counts = character_lines.groupby('movie_id').size().reset_index(name='line_count')
    
    # Merge with movie titles to get movie names
    movie_info = movie_counts.merge(titles_df, on='movie_id')
    
    # Sort by number of lines (most dialogue first)
    movie_info = movie_info.sort_values('line_count', ascending=False)
    
    print(f"\n'{character_name}' appears in {len(movie_info)} movie(s):")
    print("="*80)
    for i, row in movie_info.iterrows():
        print(f"  {row['title']:50s} ({row['year']}) - {row['line_count']:3d} lines")
    print("="*80)
    
    return movie_info

def get_character_dialogue(lines_df, titles_df, character_name, movie_title=None):
    """
    Extract all dialogue for a specific character.
    """
    # Filter to just this character (case-insensitive to catch variations)
    character_lines = lines_df[
        lines_df['character_name'].str.upper() == character_name.upper()
    ]
    if movie_title:
        # Find movie_id(s) that match the title
        matching_movies = titles_df[
            titles_df['title'].str.lower().str.contains(movie_title.lower())
        ]
        
        if len(matching_movies) == 0:
            print(f"Warning: No movie found matching '{movie_title}'")
            print("Using all movies for this character.")
        else:
            # Filter to lines from matching movie(s)
            movie_ids = matching_movies['movie_id'].tolist()
            character_lines = character_lines[
                character_lines['movie_id'].isin(movie_ids)
            ]
            
            print(f"Filtering to movie(s): {', '.join(matching_movies['title'].tolist())}")
    
    # Extract just the text of what they said
    dialogue = character_lines['text'].tolist()
    
    print(f"Found {len(dialogue)} lines of dialogue for {character_name}")
    
    return dialogue

def prepare_training_data(dialogue_list, output_file='character_dialogue.txt', min_dataset_size=500):
    """
    Prepare the dialogue for training with aggressive cleaning.
    """
    cleaned_lines = []
    
    for line in dialogue_list:
        if not line or not isinstance(line, str):
            continue
            
        # Remove only the most problematic characters, keep readable text
        line = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line)  # Remove control characters
        line = re.sub(r'[^\x20-\x7e]', '', line)  # Keep printable ASCII
        
        # Remove script artifacts
        line = re.sub(r'\[.*?\]', '', line)  # Remove stage directions
        line = re.sub(r'\(.*?\)', '', line)  # Remove parenthetical notes
        
        # Clean up whitespace and punctuation
        line = ' '.join(line.split())  # Normalize whitespace
        line = re.sub(r'([.!?]){3,}', r'\1\1', line)  # Limit repeated punctuation
        
        line = line.strip()
        
        # Keep lines that have reasonable content
        if len(line) < 3:
            continue
            
        # Check if the line has enough actual words
        words = line.split()
        if len(words) < 2:
            continue
            
        cleaned_lines.append(line)
    
    # Data augmentation for small datasets
    if len(cleaned_lines) < min_dataset_size:
        print(f"Augmenting dataset from {len(cleaned_lines)} to ~{min_dataset_size} lines...")
        augmented_lines = cleaned_lines.copy()
        
        # Create variations by combining consecutive lines
        for i in range(len(cleaned_lines) - 1):
            if random.random() < 0.5:  # 50% chance to combine
                combined = f"{cleaned_lines[i]} {cleaned_lines[i+1]}"
                if len(combined) < 200:  # Don't create overly long lines
                    augmented_lines.append(combined)
        
        # Add lines multiple times with slight variations
        while len(augmented_lines) < min_dataset_size and len(cleaned_lines) > 0:
            line = random.choice(cleaned_lines)
            augmented_lines.append(line)
            
        cleaned_lines = augmented_lines[:min_dataset_size]
    
    # Write to file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            # Add the end-of-text token for GPT-2
            f.write(f"{line}\n")
    
    print(f"Training data saved to {output_file}")
    print(f"Total lines: {len(cleaned_lines)}")
    
    
    return output_file


def setup_model_and_tokenizer():
    """
    Load the pre-trained DistilGPT-2 model and its tokenizer.
    """
    print("Loading pre-trained DistilGPT-2...")
    
    # Load the tokenizer (converts text to numbers and back)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('distilgpt2')
    
    # GPT-2 doesn't have a padding token by default, but we need one for batch training
    # We'll use the end-of-text token for padding too
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the actual model (the neural network with all its pre-trained weights)
    model = transformers.GPT2LMHeadModel.from_pretrained('distilgpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, tokenizer

def create_dataset(file_path, tokenizer, block_size=128):
    """
    Convert our text file into a format the model can train on.
    
    The TextDataset class handles reading the file, tokenizing it, and breaking
    it into chunks of the right size. The block_size parameter controls how many
    tokens each training example contains - we use 128 because it's long enough
    to capture context but short enough to train quickly.
    """
    dataset = transformers.TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size  # How many tokens per training example
    )
    
    # This data collator handles batching and creating the input/target pairs
    # For language modeling, the target is just the input shifted by one token
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # mlm=False means we're doing causal language modeling (predicting next word)
    )
    
    return dataset, data_collator

def fine_tune_model(model, tokenizer, train_dataset, data_collator, output_dir='character_model', dataset_size=None):
    """
    Fine-tune the model on our character's dialogue.
    """
    if dataset_size and dataset_size < 200:
        num_epochs = 20  # More epochs for very small datasets
        learning_rate = 2e-5  # Lower learning rate
    elif dataset_size and dataset_size < 500:
        num_epochs = 15
        learning_rate = 3e-5
    else:
        num_epochs = 10
        learning_rate = 5e-5
        
    # Training arguments control how the training process works
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,           # Where to save the fine-tuned model
        overwrite_output_dir=True,       # Overwrite if directory exists
        num_train_epochs=num_epochs,     # How many times to go through the data
        per_device_train_batch_size=2,   # How many examples to process at once
        gradient_accumulation_steps=2,   # Accumulate gradients
        save_steps=500,                  # Save a checkpoint every 500 steps
        save_total_limit=2,              # Only keep the 2 most recent checkpoints
        logging_steps=50,                # Print progress every 100 steps
        learning_rate=learning_rate,     # How fast the model learns (smaller = more careful)
        warmup_steps=100,                # Gradually increase learning rate at start
        weight_decay=0.01,               # Add weight decay for regularization
    )
    
    # The Trainer handles all the complex training loop for us
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # This is where the actual training happens
    print(f"\nStarting fine-tuning with {num_epochs} epochs...")
    trainer.train()
        
    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def generate_character_response(model, tokenizer, prompt, max_length=100):
    """
    Generate text in the character's voice.
    """
    # Convert the prompt text to token IDs
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Moving the input for the GPU used during training
    device = next(model.parameters()).device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate text
    # We use several parameters to control the generation:
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,          # Maximum total length including prompt
            num_return_sequences=1,          # Generate one response
            no_repeat_ngram_size=3,         # Don't repeat the same 2-word phrases
            do_sample=True,                  # Use sampling instead of always picking most likely
            top_k=50,                        # Only consider the top 50 most likely next tokens
            top_p=0.92,                      # Use nucleus sampling (cumulative probability)
            temperature=0.8,                 # Control randomness (higher = more random)
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    # Convert the tokens back to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the output to show only generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def interactive_generation(model, tokenizer):
    """
    Let the user have a conversation with the character model.
    """
    print("Type a prompt and see what your character says!")
    print("Type 'quit' to exit\n\n")
    
    while True:
        user_input = input("Your prompt: ")
        
        if user_input.lower() == 'quit':
            print("Quitting...\n\n")
            break
        
        print("\nGenerating response...\n")
        response = generate_character_response(model, tokenizer, user_input, max_length=60)
        print(f"Character says: {response}\n")

def main():
    """
    Run the complete project: download data, preprocess, train, and test.
    """    
    download_cornell_corpus()
    lines_df = load_movie_lines()
    titles_df = load_movie_titles()
        
    character = input("\nWhich character would you like to fine-tune on? ")
    movie_info = find_movies_for_character(lines_df, titles_df, character)
    
    if movie_info is None:
        return
    
    # If character appears in multiple movies, ask which one
    movie_title = None
    if len(movie_info) > 1:
        print("\nThis character appears in multiple movies.")
        choice = input("Enter movie title (or press Enter for ALL movies): ").strip()
        if choice:
            movie_title = choice
    dialogue = get_character_dialogue(lines_df, titles_df, character, movie_title)
    
    if len(dialogue) < 50:
        print(f"\nWarning: Only found {len(dialogue)} lines for this character.")
        return
    
    training_file = prepare_training_data(dialogue)
    
    # Step 2: Fine-tune the model
    
    model, tokenizer = setup_model_and_tokenizer()
    train_dataset, data_collator = create_dataset(training_file, tokenizer, block_size=128)
    model, tokenizer = fine_tune_model(model, tokenizer, train_dataset, data_collator, dataset_size=len(dialogue))
    
    # Step 3: Generate and inference
    
    # Try a few example generations first
    test_prompts = [
        "Hello there!",
        "What do you think about",
        "Tell me about your"
    ]
    
    for prompt in test_prompts:
        response = generate_character_response(model, tokenizer, prompt, max_length=60)
        print(f"Prompt: '{prompt}'")
        print(f"Response: {response}\n")
    
    # Then let them interact
    interactive_generation(model, tokenizer)

if __name__ == "__main__":
    main()