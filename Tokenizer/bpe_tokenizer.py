#%%
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Preprocess the text to clean and tokenize it
def preprocess_text(text):
    """Clean and prepare text for tokenization."""
    text = text.lower()  # Lowercase normalization
    text = re.sub(r"[^a-zA-Z0-9.,!?;'\s]", "", text)  # Remove special characters
    words = text.split()  # Tokenize by space
    return words


# Function to get pair frequencies in the current token list
def get_pair_frequencies(tokens):
    """Count frequency of character pairs in tokens."""
    pairs = defaultdict(int)
    for token in tokens:
        chars = token.split()
        for i in range(len(chars) - 1):
            pairs[(chars[i], chars[i + 1])] += 1
    return pairs

# Function to merge the most frequent pair and return updated tokens and frequencies
def merge_most_frequent_pair(tokens, merge_rules):
    """In a list of tokens find and merge the most frequent/best character pair."""
    frq_pairs = get_pair_frequencies(tokens)
    if not frq_pairs:
        return tokens, None, frq_pairs
    
    # Find the best pair
    best_pair = max(frq_pairs, key=frq_pairs.get)
    merge_rules.append(best_pair)
    
    # Merge the best pair in the tokens
    updated_tokens = []
    for token in tokens:
        updated_tokens.append(token.replace(" ".join(best_pair), "".join(best_pair)))
    
    return updated_tokens, best_pair, frq_pairs



def create_animation(token_states, merge_rules):
    # Create the animation
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(frame):
        ax.clear()

        # Get previous and current tokens
        prev_tokens = " ".join(token_states[frame])
        new_tokens = " ".join(token_states[frame + 1]) if frame + 1 < len(token_states) else ""

        # AI: Split the previous and updated tokens into separate lines
        prev_tokens_split = "\n".join(token_states[frame])
        new_tokens_split = "\n".join(token_states[frame + 1]) if frame + 1 < len(token_states) else ""

        # AI: Bold the merged pair in the updated tokens
        merged_pair = merge_rules[frame] if frame < len(merge_rules) else None
        bold_prev_tokens = []
        bold_new_tokens = []

        if merged_pair:
            # Bold the previous tokens (before merge)
            for token in token_states[frame]:
                if " ".join(merged_pair) in token:
                    bold_prev_tokens.append(f"**{token}**")   # Make bold
                else:
                    bold_prev_tokens.append(token)

            # Bold the new tokens (after merge)
            for token in token_states[frame + 1]:
                if " ".join(merged_pair) in token:
                    bold_new_tokens.append(f"**{token}**")  # Make bold
                else:
                    bold_new_tokens.append(token)
        else:
            # If no merge, just show regular tokens
            bold_prev_tokens = token_states[frame]
            bold_new_tokens = token_states[frame + 1] if frame + 1 < len(token_states) else []

        #Display the previous tokens in the left box with bold changes
        prev_text = f"Before Merge:\n" + "\n".join(bold_prev_tokens)
        ax.text(0.35, 0.5, prev_text, fontsize=12, ha='center', va='center', 
                bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

        #Display the updated tokens in the right box with bold changes
        new_text = f"After Merge:\n" + "\n".join(bold_new_tokens)
        ax.text(1.1, 0.5, new_text, fontsize=12, ha='center', va='center', 
                bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

        #Adjust the axis limits and remove ticks for cleaner layout
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"BPE Step {frame + 1}")
        
        #Set the plot limits to ensure both text boxes are displayed side by side properly
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1)


    ani = animation.FuncAnimation(fig, update, frames=len(token_states) - 1, repeat=False)

    # Save the animation as a GIF
    ani.save("./bpe_animation_with_legend.gif", writer="pillow", fps=1)
    print("Animation saved as bpe_animation_with_legend.gif")


def apply_bpe_tokenization(text, merge_rules):
    """Tokenize a new input using learned BPE rules."""
    #1. Preprocess the unknown the text
    words = preprocess_text(text)
    tokens = [" ".join(word) + " </w>" for word in words]

    #2. Use the trained merged rules to tokenize the data
    for merge_pair in merge_rules:
        updated_tokens = []
        for token in tokens:
            updated_tokens.append(
                token.replace(" ".join(merge_pair), "".join(merge_pair))
            )
        tokens = updated_tokens
    return tokens


def detokenize(tokens):
    """Reconstruct original text from tokenized output."""
    text = " ".join(tokens)  # Join tokens back into a string
    text = text.replace(" </w>", "")  # Remove word boundary markers
    return text

if __name__ == "__main__":
    # Sample text corpus
    text_corpus = "Hello world! This is a small test dataset for training a tokenizer."
    words = preprocess_text(text_corpus)

    # Initialize character-level tokens (space-separated characters)
    tokens = [" ".join(word) + " </w>" for word in words]

    
    # Training loop: Merge the top N frequent pairs
    num_merges = 10  # Limit merges for testing
    merge_rules = []
    token_states = [tokens.copy()]
    pair_frequencies = []

    for _ in range(num_merges):
        tokens, merged_pair, frq_pairs = merge_most_frequent_pair(tokens, merge_rules)
        pair_frequencies.append(frq_pairs)
        if merged_pair is None:
            break  # Stop if no more merges
        token_states.append(tokens.copy())