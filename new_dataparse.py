import json
import os
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

class DiscordDataFormatter:
    def __init__(self, input_directory: str, output_file: str, bot_token: Optional[str] = None):
        self.input_directory = input_directory
        self.output_file = output_file
        self.bot_token = bot_token
        self.user_map = {}  # Map authorId to readable names
        self.bot = None
        self.use_bot = bot_token is not None
        
    async def setup_bot(self):
        """Set up the Discord bot for fetching usernames"""
        if not self.use_bot:
            return
            
        # Set up bot with minimal intents
        intents = discord.Intents.default()
        intents.message_content = False  # We don't need message content
        intents.guilds = True  # We need guild access to fetch users
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        
        # Create an event to wait for bot ready
        self.bot_ready = asyncio.Event()
        
        @self.bot.event
        async def on_ready():
            print(f'Bot logged in as {self.bot.user}')
            self.bot_ready.set()  # Signal that bot is ready
        
        try:
            # Start the bot connection in the background
            bot_task = asyncio.create_task(self.bot.start(self.bot_token))
            
            # Wait for bot to be ready with timeout
            try:
                await asyncio.wait_for(self.bot_ready.wait(), timeout=30.0)
                print("Bot is ready!")
            except asyncio.TimeoutError:
                print("Bot connection timed out")
                bot_task.cancel()
                self.use_bot = False
                return
                
        except Exception as e:
            print(f"Failed to connect bot: {e}")
            self.use_bot = False
    
    async def fetch_username(self, user_id: str) -> Optional[str]:
        """Fetch username from Discord API using bot"""
        if not self.use_bot or not self.bot:
            return None
            
        try:
            user_id_int = int(user_id)
            user = await self.bot.fetch_user(user_id_int)
            return user.display_name or user.name
        except Exception as e:
            print(f"Failed to fetch user {user_id}: {e}")
            return None
    
    async def populate_user_map(self, user_ids: set):
        """Populate the user map with real usernames from Discord"""
        if not self.use_bot:
            print("Bot not available, using anonymous usernames")
            return
            
        print(f"Fetching usernames for {len(user_ids)} users...")
        
        for i, user_id in enumerate(user_ids, 1):
            if user_id in self.user_map:
                continue
                
            print(f"  Fetching user {i}/{len(user_ids)}: {user_id}")
            
            username = await self.fetch_username(user_id)
            if username:
                self.user_map[user_id] = username
                print(f"    -> {username}")
            else:
                # Fallback to anonymous username
                user_num = len(self.user_map) + 1
                self.user_map[user_id] = f"User{user_num}"
                print(f"    -> User{user_num} (fallback)")
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.1)
    
    async def cleanup_bot(self):
        """Clean up bot connection"""
        if self.bot and not self.bot.is_closed():
            print("Closing bot connection...")
            await self.bot.close()
            # Wait a bit for cleanup
            await asyncio.sleep(1)
        
    def clean_content(self, content: str) -> str:
        """Clean message content by removing URLs, mentions, and other unwanted elements"""
        if not content:
            return ""
            
        # Remove URLs
        content = re.sub(r'https?://[^\s]+', '', content)
        content = re.sub(r'www\.[^\s]+', '', content)
        
        # Remove Discord mentions (@user, @everyone, @here)
        content = re.sub(r'@\w+', '', content)
        content = re.sub(r'@everyone', '', content)
        content = re.sub(r'@here', '', content)
        
        # Remove Discord channel mentions (#channel)
        content = re.sub(r'#\w+', '', content)
        
        # Remove Discord emojis/reactions (<:emoji:id>)
        content = re.sub(r'<:[^:]+:\d+>', '', content)
        content = re.sub(r'<a:[^:]+:\d+>', '', content)
        
        # Remove role mentions (<@&roleId>)
        content = re.sub(r'<@&\d+>', '', content)
        
        # Remove user mentions (<@userId>)
        content = re.sub(r'<@!?\d+>', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def should_skip_message(self, message: Dict[str, Any]) -> bool:
        """Determine if a message should be skipped"""
        content = message.get('content', '')
        
        # Skip deleted messages
        if message.get('deleted', False):
            return True
            
        # Skip empty messages
        if not content or not content.strip():
            return True
            
        # Skip messages that are only URLs
        if re.match(r'^https?://[^\s]+$', content.strip()):
            return True
            
        # Skip messages that are only mentions
        if re.match(r'^@\w+$', content.strip()):
            return True
            
        # Skip very short messages (less than 3 characters after cleaning)
        cleaned = self.clean_content(content)
        if len(cleaned) < 3:
            return True
            
        # Skip messages with only special characters
        if re.match(r'^[^a-zA-Z0-9\s]*$', cleaned):
            return True
            
        return False
    
    def get_username(self, author_id: str) -> str:
        """Get a readable username for an author ID"""
        if author_id not in self.user_map:
            # Create anonymous username as fallback
            user_num = len(self.user_map) + 1
            self.user_map[author_id] = f"User{user_num}"
        return self.user_map[author_id]
    
    def collect_user_ids(self) -> set:
        """Collect all unique user IDs from all channel files"""
        user_ids = set()
        
        json_files = [f for f in os.listdir(self.input_directory) if f.endswith('.json')]
        
        for filename in json_files:
            file_path = os.path.join(self.input_directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                messages = data.get('messages', [])
                for message in messages:
                    if not self.should_skip_message(message):
                        user_ids.add(message['authorId'])
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        return user_ids
    
    def process_channel_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single channel JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
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
        """Process all JSON files in the input directory"""
        all_messages = []
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.input_directory) if f.endswith('.json')]
        
        print(f"Found {len(json_files)} channel files")
        
        for filename in json_files:
            file_path = os.path.join(self.input_directory, filename)
            print(f"Processing {filename}...")
            
            channel_messages = self.process_channel_file(file_path)
            all_messages.extend(channel_messages)
            
            print(f"  Added {len(channel_messages)} messages from {filename}")
        
        # Sort all messages by timestamp
        all_messages.sort(key=lambda x: x['timestamp'])
        
        return all_messages
    
    def format_for_training(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format messages for training with conversation context"""
        training_data = []
        context_window = 3  # Number of previous messages to include as context
        
        for i in range(len(messages)):
            # Get context messages (previous messages)
            context_start = max(0, i - context_window)
            context_messages = messages[context_start:i]
            
            # Format context
            context_parts = []
            for msg in context_messages:
                context_parts.append(f"{msg['username']}: {msg['message']}")
            
            # Current message
            current_msg = messages[i]
            target = f"{current_msg['username']}: {current_msg['message']}"
            
            # Create training example
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
        """Save the formatted training data to a JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(training_data)} training examples to {self.output_file}")
    
    def save_user_mapping(self):
        """Save the user ID to username mapping for reference"""
        mapping_file = f"{self.output_file.rsplit('.', 1)[0]}_user_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_map, f, indent=2, ensure_ascii=False)
        print(f"Saved user mapping to {mapping_file}")
    
    async def run(self):
        """Run the complete formatting process"""
        print("Starting Discord data formatting...")
        
        bot_task = None
        try:
            # Set up bot if token provided
            if self.use_bot:
                await self.setup_bot()
            
            # Only proceed with username fetching if bot is ready
            if self.use_bot and self.bot and self.bot.is_ready():
                print("Collecting user IDs...")
                user_ids = self.collect_user_ids()
                print(f"Found {len(user_ids)} unique users")
                
                # Fetch real usernames
                await self.populate_user_map(user_ids)
            else:
                print("Bot not available or not ready, using anonymous usernames")
            
            # Process all channel files
            all_messages = self.process_all_channels()
            print(f"Total messages processed: {len(all_messages)}")
            
            if not all_messages:
                print("No messages found! Check your input directory and files.")
                return
            
            # Format for training
            training_data = self.format_for_training(all_messages)
            
            # Save the results
            self.save_training_data(training_data)
            self.save_user_mapping()
            
            print("Formatting complete!")
            print(f"User mapping: {len(self.user_map)} unique users")
            
            # Show sample of what was created
            if training_data:
                print("\nSample training example:")
                print("-" * 50)
                print(training_data[0]['text'][:200] + "..." if len(training_data[0]['text']) > 200 else training_data[0]['text'])
                print("-" * 50)
                
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up bot connection
            await self.cleanup_bot()

# Synchronous wrapper for backwards compatibility
def run_formatter_sync(input_directory: str, output_file: str, bot_token: Optional[str] = None):
    """Synchronous wrapper to run the formatter"""
    formatter = DiscordDataFormatter(input_directory, output_file, bot_token)
    asyncio.run(formatter.run())

# Usage
if __name__ == "__main__":
    # Configure paths
    input_directory = "realdata"  # Directory containing your JSON files
    output_file = "discord_training_data.json"
    
    # Optional: Add your Discord bot token here
    # You can get a bot token from https://discord.com/developers/applications
    bot_token = os.getenv('DISCORD_BOT_TOKEN')  # Replace with your bot token: "YOUR_BOT_TOKEN_HERE"
    
    # Create formatter and run
    run_formatter_sync(input_directory, output_file, bot_token)