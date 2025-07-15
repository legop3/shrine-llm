import json
import os
import re
from datetime import datetime
from collections import defaultdict

def clean_message_content(content):
    """Clean message content by removing URLs, custom emojis, and user pings"""
    # Remove URLs (http, https, www)
    content = re.sub(r'https?://[^\s]+', '', content)
    content = re.sub(r'www\.[^\s]+', '', content)
    
    # Remove custom emoji IDs (format: <:emoji_name:123456789>)
    content = re.sub(r'<:[^:]+:\d+>', '', content)
    
    # Remove animated custom emojis (format: <a:emoji_name:123456789>)
    content = re.sub(r'<a:[^:]+:\d+>', '', content)
    
    # Remove user pings (format: <@123456789> or <@!123456789>)
    content = re.sub(r'<@!?\d+>', '', content)
    
    # Remove role pings (format: <@&123456789>)
    content = re.sub(r'<@&\d+>', '', content)
    
    # Remove channel references (format: <#123456789>)
    content = re.sub(r'<#\d+>', '', content)
    
    # Clean up extra whitespace
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    return content

def has_embeds_or_attachments(message):
    """Check if message has embeds or attachments"""
    # Check for embeds
    if message.get('embeds') and len(message['embeds']) > 0:
        return True
    
    # Check for attachments
    if message.get('attachments') and len(message['attachments']) > 0:
        return True
    
    return False

def load_discord_data(data_folder):
    """Load all channel JSON files and combine messages"""
    all_messages = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            print(f"Processing file: {filename}")
            try:
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
                    channel_data = json.load(f)
                
                # Handle different JSON structures
                if 'messages' in channel_data:
                    # Your original format
                    messages = channel_data['messages']
                    channel_name = channel_data.get('name', filename.replace('.json', ''))
                elif isinstance(channel_data, list):
                    # If it's just a list of messages
                    messages = channel_data
                    channel_name = filename.replace('.json', '')
                else:
                    print(f"Unknown format in {filename}, skipping...")
                    continue
                
                # Add channel info to each message
                for message in messages:
                    # Make sure message has required fields
                    if not isinstance(message, dict):
                        continue
                    if 'content' not in message:
                        continue
                    if 'authorId' not in message:
                        continue
                    
                    message['channelName'] = channel_name
                    all_messages.append(message)
                    
                print(f"  Loaded {len(messages)} messages from {channel_name}")
                
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Sort by timestamp
    all_messages.sort(key=lambda x: x.get('createdAt', ''))
    return all_messages

def group_consecutive_messages(messages):
    """Group consecutive messages from the same author"""
    grouped = []
    current_group = []
    current_author = None
    
    for msg in messages:
        # Skip deleted messages
        if msg.get('deleted', False):
            continue
        
        # Skip messages with embeds or attachments
        if has_embeds_or_attachments(msg):
            continue
        
        # Skip messages without content
        if not msg.get('content'):
            continue
        
        # Clean the message content
        cleaned_content = clean_message_content(msg['content'])
        
        # Skip if cleaned content is too short or empty
        if len(cleaned_content.strip()) < 3:
            continue
        
        # Skip messages that are mostly just links/emojis/pings (now empty after cleaning)
        if not cleaned_content.strip():
            continue
            
        author = msg['authorId']
        
        if author == current_author:
            # Same author, add to current group
            msg['cleaned_content'] = cleaned_content
            current_group.append(msg)
        else:
            # New author, save previous group and start new one
            if current_group:
                grouped.append(current_group)
            msg['cleaned_content'] = cleaned_content
            current_group = [msg]
            current_author = author
    
    # Don't forget the last group
    if current_group:
        grouped.append(current_group)
    
    return grouped

def create_training_pairs(grouped_messages):
    """Create input/response pairs from grouped messages"""
    training_pairs = []
    
    for i in range(len(grouped_messages) - 1):
        input_group = grouped_messages[i]
        response_group = grouped_messages[i + 1]
        
        # Combine cleaned messages in each group
        input_text = ' '.join([msg['cleaned_content'] for msg in input_group])
        response_text = ' '.join([msg['cleaned_content'] for msg in response_group])
        
        # Final cleanup
        input_text = input_text.strip()
        response_text = response_text.strip()
        
        # Skip if either is too short or too long
        if len(input_text) < 5 or len(response_text) < 5:
            continue
        if len(input_text) > 500 or len(response_text) > 500:
            continue
        
        # Skip if content is mostly punctuation or special characters
        if len(re.sub(r'[^\w\s]', '', input_text)) < 3:
            continue
        if len(re.sub(r'[^\w\s]', '', response_text)) < 3:
            continue
            
        training_pairs.append({
            'input': input_text,
            'response': response_text
        })
    
    return training_pairs

def main():
    # Load data
    data_folder = './realdata'  # Change this to your folder
    messages = load_discord_data(data_folder)
    
    print(f"Loaded {len(messages)} total messages")
    
    # Group consecutive messages
    grouped = group_consecutive_messages(messages)
    print(f"Created {len(grouped)} message groups")
    
    # Create training pairs
    pairs = create_training_pairs(grouped)
    print(f"Created {len(pairs)} training pairs")
    
    # Save to JSON
    with open('discord_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    print("Saved training data to discord_training_data.json")
    
    # Show some examples
    print("\nFirst 3 examples:")
    for i, pair in enumerate(pairs[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {pair['input']}")
        print(f"Response: {pair['response']}")
    
    # Show filtering statistics
    total_messages = len(messages)
    filtered_messages = len([msg for msg in messages if not msg.get('deleted', False) and not has_embeds_or_attachments(msg)])
    print(f"\nFiltering stats:")
    print(f"Total messages: {total_messages}")
    print(f"After filtering: {filtered_messages}")
    print(f"Filtered out: {total_messages - filtered_messages} ({((total_messages - filtered_messages) / total_messages * 100):.1f}%)")

if __name__ == "__main__":
    main()