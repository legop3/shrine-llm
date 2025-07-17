import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
import os

@dataclass
class DiscordMessage:
    message_id: str
    user_id: str
    username: str
    content: str
    timestamp: datetime
    channel_id: str
    channel_name: str
    reply_to: str = None
    context_messages: List[Dict] = None

class DiscordRAGSystem:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 db_path: str = "./discord_embeddings"):
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB for vector storage
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="discord_messages",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize generation model (you can swap this for larger models)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # User mapping for better display names
        self.user_mapping = {}
    
    def load_user_mapping(self, user_mapping_file: str = None):
        """
        Load user ID to username mapping from file
        Format: {"user_id": "username", ...}
        """
        if user_mapping_file and os.path.exists(user_mapping_file):
            with open(user_mapping_file, 'r', encoding='utf-8') as f:
                self.user_mapping = json.load(f)
        
    def get_username(self, user_id: str) -> str:
        """Get username from user ID, fallback to truncated ID"""
        return self.user_mapping.get(user_id, f"User_{user_id[-8:]}")
    
    def create_user_mapping_from_data(self, discord_folder_path: str):
        """
        Create a basic user mapping from the data itself
        You can manually edit this afterward
        """
        import os
        user_ids = set()
        
        json_files = [f for f in os.listdir(discord_folder_path) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(discord_folder_path, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    channel_data = json.load(f)
                
                for msg_data in channel_data.get('messages', []):
                    if not msg_data.get('deleted', False) and msg_data.get('content'):
                        user_ids.add(msg_data['authorId'])
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Create basic mapping
        mapping = {}
        for user_id in user_ids:
            mapping[user_id] = f"User_{user_id[-8:]}"
        
        # Save mapping file for manual editing
        with open('user_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Created user_mapping.json with {len(mapping)} users")
        print("You can edit this file to add real usernames!")
        
        return mapping
    
    def preprocess_discord_data(self, discord_folder_path: str) -> List[DiscordMessage]:
        """
        Process Discord data export from multiple channel JSON files
        """
        import os
        messages = []
        
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(discord_folder_path) if f.endswith('.json')]
        
        print(f"Found {len(json_files)} channel files")
        
        for json_file in json_files:
            file_path = os.path.join(discord_folder_path, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    channel_data = json.load(f)
                
                channel_id = channel_data.get('channelId')
                channel_name = channel_data.get('name', 'unknown')
                
                print(f"Processing channel: {channel_name} ({len(channel_data.get('messages', []))} messages)")
                
                for msg_data in channel_data.get('messages', []):
                    # Skip deleted messages or messages without content
                    if msg_data.get('deleted', False) or not msg_data.get('content'):
                        continue
                    
                    message = DiscordMessage(
                        message_id=msg_data['id'],
                        user_id=msg_data['authorId'],
                        username=self.get_username(msg_data['authorId']),
                        content=msg_data['content'],
                        timestamp=datetime.fromisoformat(msg_data['createdAt'].replace('Z', '+00:00')),
                        channel_id=channel_id,
                        channel_name=channel_name,
                        reply_to=None  # You can add reply detection logic if needed
                    )
                    messages.append(message)
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        return messages
    
    def create_conversation_context(self, messages: List[DiscordMessage]) -> List[Dict]:
        """
        Group messages into conversation threads with context
        """
        conversations = []
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda x: x.timestamp)
        
        # Group by channel and time windows (e.g., 10 minutes)
        current_convo = []
        last_timestamp = None
        last_channel = None
        
        for msg in sorted_messages:
            # Start new conversation if channel changes or time gap > 10 minutes
            if (last_channel != msg.channel_id or 
                (last_timestamp and (msg.timestamp - last_timestamp).seconds > 600)):
                
                if current_convo:
                    conversations.append(self.format_conversation(current_convo))
                current_convo = [msg]
            else:
                current_convo.append(msg)
            
            last_timestamp = msg.timestamp
            last_channel = msg.channel_id
        
        if current_convo:
            conversations.append(self.format_conversation(current_convo))
        
        return conversations
    
    def format_conversation(self, messages: List[DiscordMessage]) -> Dict:
        """
        Format a conversation thread for embedding and storage
        """
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(f"{msg.username}: {msg.content}")
        
        conversation_text = "\n".join(formatted_messages)
        
        return {
            "text": conversation_text,
            "metadata": {
                "channel": messages[0].channel_name,
                "start_time": messages[0].timestamp.isoformat(),
                "end_time": messages[-1].timestamp.isoformat(),
                "participants": list(set([msg.username for msg in messages])),
                "message_count": len(messages)
            }
        }
    
    def build_vector_database(self, discord_folder_path: str):
        """
        Build the vector database from Discord folder containing channel JSON files
        """
        print("Processing Discord messages...")
        messages = self.preprocess_discord_data(discord_folder_path)
        
        print(f"Found {len(messages)} messages across all channels")
        
        print("Creating conversation contexts...")
        conversations = self.create_conversation_context(messages)
        
        print(f"Created {len(conversations)} conversation threads")
        
        # Create embeddings and store in ChromaDB
        print("Creating embeddings...")
        texts = [conv["text"] for conv in conversations]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Prepare data for ChromaDB
        ids = [f"conv_{i}" for i in range(len(conversations))]
        metadatas = [conv["metadata"] for conv in conversations]
        
        # Add to collection in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(conversations), batch_size):
            end_idx = min(i + batch_size, len(conversations))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(conversations) + batch_size - 1)//batch_size}")
        
        print(f"Added {len(conversations)} conversations to vector database")
    
    def retrieve_relevant_context(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Retrieve relevant conversations based on query
        """
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def generate_response(self, query: str, max_length: int = 200) -> str:
        """
        Generate response using retrieved context
        """
        # Get relevant context
        context_results = self.retrieve_relevant_context(query)
        
        # Build context prompt
        context_text = ""
        for i, doc in enumerate(context_results['documents'][0]):
            context_text += f"Context {i+1}:\n{doc}\n\n"
        
        # Create prompt in a conversational format
        prompt = f"""Based on the following conversations from our Discord server, respond as a member of this community would:

{context_text}

Current message: {query}
Response:"""
        
        # Generate response
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        response = response.split("Response:")[-1].strip()
        
        return response

# Usage example
def main():
    # Initialize the system
    rag_system = DiscordRAGSystem()
    
    discord_folder = "./discord_channels"  # Update this path to your folder
    
    # Optional: Create user mapping file (run once, then edit manually)
    # rag_system.create_user_mapping_from_data(discord_folder)
    
    # Load user mapping if available
    rag_system.load_user_mapping("user_mapping.json")
    
    # Build vector database from Discord folder
    rag_system.build_vector_database(discord_folder)
    
    # Interactive chat loop
    print("Discord RAG Bot initialized! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = rag_system.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()