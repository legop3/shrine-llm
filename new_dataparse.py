import json
import os
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import discord
from discord.ext import commands

class DiscordDataFormatter:
    def __init__(self, input_directory: str, output_file: str, bot_token: Optional[str] = None, 
                 filter_user: Optional[str] = None):
        self.input_directory = input_directory
        self.output_file = output_file
        self.bot_token = bot_token
        self.filter_user = filter_user
        self.filter_user_id = None
        self.user_map = {}
        self.bot = None
        self.use_bot = bot_token is not None
        
    async def setup_bot(self):
        """Set up the Discord bot for fetching usernames"""
        if not self.use_bot:
            return False
            
        try:
            intents = discord.Intents.default()
            intents.message_content = False
            intents.guilds = True
            
            self.bot = commands.Bot(command_prefix='!', intents=intents)
            
            @self.bot.event
            async def on_ready():
                print(f'Bot logged in as {self.bot.user}')
            
            await self.bot.start(self.bot_token)
            
            # Wait for bot to be ready
            while not self.bot.is_ready():
                await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            print(f"Failed to connect bot: {e}")
            self.use_bot = False
            return False
    
    async def fetch_username(self, user_id: str) -> Optional[str]:
        """Fetch username from Discord API"""
        if not self.use_bot or not self.bot:
            return None
            
        try:
            user = await self.bot.fetch_user(int(user_id))
            return user.display_name or user.name
        except Exception as e:
            print(f"Failed to fetch user {user_id}: {e}")
            return None
    
    def clean_content(self, content: str) -> str:
        """Clean message content"""
        if not content:
            return ""
            
        # Remove URLs
        content = re.sub(r'https?://[^\s]+', '', content)
        content = re.sub(r'www\.[^\s]+', '', content)
        
        # Remove Discord mentions and emojis
        content = re.sub(r'@\w+', '', content)
        content = re.sub(r'@everyone', '', content)
        content = re.sub(r'@here', '', content)
        content = re.sub(r'#\w+', '', content)
        content = re.sub(r'<:[^:]+:\d+>', '', content)
        content = re.sub(r'<a:[^:]+:\d+>', '', content)
        content = re.sub(r'<@&\d+>', '', content)
        content = re.sub(r'<@!?\d+>', '', content)
        
        # Clean whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def should_skip_message(self, message: Dict[str, Any]) -> bool:
        """Determine if a message should be skipped"""
        content = message.get('content', '')
        author_id = message.get('authorId', '')
        
        # Skip if user filtering is enabled and this isn't the target user
        if self.filter_user_id and author_id != self.filter_user_id:
            return True
        
        # Skip deleted messages
        if message.get('deleted', False):
            return True
            
        # Skip empty messages
        if not content or not content.strip():
            return True
            
        # Skip very short messages after cleaning
        cleaned = self.clean_content(content)
        if len(cleaned) < 3:
            return True
            
        return False
    
    def get_username(self, author_id: str) -> str:
        """Get username for author ID"""
        if author_id not in self.user_map:
            user_num = len(self.user_map) + 1
            self.user_map[author_id] = f"User{user_num}"
        return self.user_map[author_id]
    
    def collect_all_user_ids(self) -> set:
        """Collect all unique user IDs from all files"""
        user_ids = set()
        json_files = [f for f in os.listdir(self.input_directory) if f.endswith('.json')]
        
        for filename in json_files:
            file_path = os.path.join(self.input_directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                messages = data.get('messages', [])
                for message in messages:
                    if not message.get('deleted', False) and message.get('content', '').strip():
                        user_ids.add(message['authorId'])
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        return user_ids
    
    async def populate_usernames(self, user_ids: set):
        """Fetch usernames for all user IDs"""
        if not self.use_bot or not self.bot:
            print("Bot not available, using anonymous usernames")
            return
            
        print(f"Fetching usernames for {len(user_ids)} users...")
        
        for i, user_id in enumerate(user_ids, 1):
            print(f"  Fetching {i}/{len(user_ids)}: {user_id}")
            
            username = await self.fetch_username(user_id)
            if username:
                self.user_map[user_id] = username
                print(f"    -> {username}")
            else:
                user_num = len(self.user_map) + 1
                self.user_map[user_id] = f"User{user_num}"
                print(f"    -> User{user_num} (fallback)")
            
            await asyncio.sleep(0.1)  # Rate limiting
    
    def resolve_filter_user(self):
        """Resolve filter user to user ID"""
        if not self.filter_user:
            return
            
        # If it's already a user ID
        if self.filter_user.isdigit():
            self.filter_user_id = self.filter_user
            print(f"Using user ID filter: {self.filter_user_id}")
            return
        
        # Search by username
        for user_id, username in self.user_map.items():
            if username.lower() == self.filter_user.lower():
                self.filter_user_id = user_id
                print(f"Found user '{username}' with ID: {user_id}")
                return
        
        # Not found
        print(f"User '{self.filter_user}' not found. Available users:")
        for user_id, username in self.user_map.items():
            print(f"  - {username} (ID: {user_id})")
        
        # Ask for user ID
        try:
            response = input(f"Enter user ID for '{self.filter_user}' or press Enter to disable filtering: ")
            if response.strip():
                self.filter_user_id = response.strip()
                print(f"Using user ID: {self.filter_user_id}")
            else:
                print("User filtering disabled")
                self.filter_user = None
        except KeyboardInterrupt:
            print("\nUser filtering disabled")
            self.filter_user = None
    
    def process_channel_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single channel file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        channel_name = data.get('name', 'unknown')
        messages = data.get('messages', [])
        processed_messages = []
        
        for message in messages:
            if self.should_skip_message(message):
                continue
                
            content = self.clean_content(message['content'])
            if not content:
                continue
                
            username = self.get_username(message['authorId'])
            timestamp = message.get('createdAt', '')
            
            processed_messages.append({
                'username': username,
                'message': content,
                'timestamp': timestamp,
                'channel': channel_name
            })
        
        return processed_messages
    
    def process_all_channels(self) -> List[Dict[str, Any]]:
        """Process all channel files"""
        all_messages = []
        json_files = [f for f in os.listdir(self.input_directory) if f.endswith('.json')]
        
        print(f"Found {len(json_files)} channel files")
        
        for filename in json_files:
            file_path = os.path.join(self.input_directory, filename)
            print(f"Processing {filename}...")
            
            channel_messages = self.process_channel_file(file_path)
            all_messages.extend(channel_messages)
            
            print(f"  Added {len(channel_messages)} messages")
        
        # Sort by timestamp
        all_messages.sort(key=lambda x: x['timestamp'])
        return all_messages
    
    def format_for_training(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format messages for training"""
        training_data = []
        context_window = 0
        
        for i in range(len(messages)):
            context_start = max(0, i - context_window)
            context_messages = messages[context_start:i]
            
            context_parts = []
            for msg in context_messages:
                context_parts.append(f"{msg['username']}: {msg['message']}")
            
            current_msg = messages[i]
            target = f"{current_msg['username']}: {current_msg['message']}"
            
            if context_parts:
                full_context = "\n".join(context_parts)
                training_example = f"{full_context}\n{target}"
            else:
                training_example = target
            
            training_data.append({
                "text": training_example,
                "channel": current_msg['channel'],
                "timestamp": current_msg['timestamp']
            })
        
        return training_data
    
    def save_training_data(self, training_data: List[Dict[str, str]]):
        """Save training data to file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(training_data)} training examples to {self.output_file}")
    
    def save_user_mapping(self):
        """Save user mapping to file"""
        mapping_file = f"{self.output_file.rsplit('.', 1)[0]}_user_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_map, f, indent=2, ensure_ascii=False)
        print(f"Saved user mapping to {mapping_file}")
    
    async def run(self):
        """Run the complete formatting process"""
        print("Starting Discord data formatting...")
        
        if self.filter_user:
            print(f"User filtering enabled for: {self.filter_user}")
        
        try:
            # Setup bot if needed
            if self.use_bot:
                success = await self.setup_bot()
                if not success:
                    print("Continuing without bot...")
            
            # Collect all user IDs
            print("Collecting user IDs...")
            user_ids = self.collect_all_user_ids()
            print(f"Found {len(user_ids)} unique users")
            
            # Fetch usernames
            await self.populate_usernames(user_ids)
            
            # Resolve filter user
            if self.filter_user:
                self.resolve_filter_user()
            
            # Process messages
            all_messages = self.process_all_channels()
            print(f"Total messages processed: {len(all_messages)}")
            
            if self.filter_user_id:
                filter_username = self.get_username(self.filter_user_id)
                filtered_count = len(all_messages)
                print(f"Messages from '{filter_username}': {filtered_count}")
            
            if not all_messages:
                print("No messages found!")
                return
            
            # Format and save
            training_data = self.format_for_training(all_messages)
            self.save_training_data(training_data)
            self.save_user_mapping()
            
            print("Formatting complete!")
            print(f"User mapping: {len(self.user_map)} unique users")
            
            if training_data:
                print("\nSample training example:")
                print("-" * 50)
                sample = training_data[0]['text']
                print(sample[:200] + "..." if len(sample) > 200 else sample)
                print("-" * 50)
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.bot:
                await self.bot.close()

def run_formatter_sync(input_directory: str, output_file: str, bot_token: Optional[str] = None, 
                      filter_user: Optional[str] = None):
    """Run the formatter synchronously"""
    formatter = DiscordDataFormatter(input_directory, output_file, bot_token, filter_user)
    asyncio.run(formatter.run())

# Usage
if __name__ == "__main__":
    input_directory = "realdata"
    output_file = "discord_training_data.json"
    bot_token = os.getenv('DISCORD_BOT_TOKEN')  # "YOUR_BOT_TOKEN_HERE"
    filter_user = "383320447514574848"  # "username" or "123456789"
    
    run_formatter_sync(input_directory, output_file, bot_token, filter_user)