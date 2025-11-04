import os
import random
import asyncio
import discord
from discord.ext import commands
from openai import ChatCompletion
import aiohttp
import json
import psycopg2
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class FunCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(help="Guess a random number between 1 and 10.")
    async def guess(self, ctx, number: int):
        random_number = random.randint(1, 10)
        if number == random_number:
            await ctx.send(f"Congratulations, {ctx.author.name}! You guessed the number! It was {random_number}!")
        else:
            await ctx.send(f"Sorry, {ctx.author.name}. The correct number was {random_number}. Better luck next time!")

    @commands.command()
    async def ping(self, ctx):
        await ctx.send('Pong!')

class AdminCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(help="Warn a user.")
    async def warn(self, ctx, member: discord.Member, *, reason=None):
        warning_message = f"You have been warned by {ctx.message.author.name}: {reason}"
        await member.send(warning_message)

    @commands.command(help="Announce a message.")
    async def announce(self, ctx, *, announcement):
        channel = discord.utils.get(ctx.guild.channels, name='general') 
        await channel.send(announcement)

    @commands.command(help="Resolve an issue.")
    async def resolve(self, ctx, *, issue):
        resolution = f"The issue: '{issue}' is now resolved. Please contact the moderators for further clarification."
        await ctx.send(resolution)

class ResourceCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        try:
            self.conn = psycopg2.connect(
                dbname="resource_links",
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host="localhost"
            )
        except psycopg2.OperationalError as e:
            print(f"Could not connect to PostgreSQL database: {e}")
            self.conn = None

    @commands.command(help="Send a resource to a user.")
    async def sendResource(self, ctx, member: discord.Member):
        if not self.conn:
            await ctx.send("Database connection not available. Please check configuration.")
            return

        resource_channel = discord.utils.get(ctx.guild.channels, name='resources')
        if not resource_channel:
            await ctx.send("Resources channel not found.")
            return

        try:
            resource_messages = [m async for m in resource_channel.history(limit=10)]

            resource_links = [m.content for m in resource_messages if m.content.startswith('http')]

            with self.conn.cursor() as cur:
                for link in resource_links:
                    cur.execute("INSERT INTO resources (link) VALUES (%s) ON CONFLICT (link) DO NOTHING", (link,))
            self.conn.commit()

            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:5001/get_resource_data', timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                    else:
                        await ctx.send(f"Failed to fetch resource data: HTTP {response.status}")
                        return

            for resource in data.values():
                report = f"Title: {resource['title']}\nLink: {resource['link']}\nDescription: {resource['description']}"
            await member.send(report)
            await ctx.send(f"Resources sent to {member.mention}")
        except aiohttp.ClientError as e:
            await ctx.send(f"Network error: {str(e)}")
        except Exception as e:
            await ctx.send(f"Error sending resources: {str(e)}")
class AppCommand(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(help="Use the AI research agent app.")
    async def app(self, ctx, *, query):
        """Send the query to the FastAPI server running the app.py code."""
        try:
            async with aiohttp.ClientSession() as session:
                api_url = "http://localhost:8000/"
                headers = {"Content-Type": "application/json"}
                data = {"query": query}
                timeout = aiohttp.ClientTimeout(total=120)
                async with session.post(api_url, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                    if response.status == 200:
                        result = await response.text()
                        await ctx.send(result)
                    else:
                        await ctx.send(f"Error: API returned status {response.status}")
        except aiohttp.ClientError as e:
            await ctx.send(f"Network error connecting to research agent: {str(e)}")
        except asyncio.TimeoutError:
            await ctx.send("Research request timed out. Try a simpler query.")

class DiscordBot(commands.Bot):
    def __init__(self, token: str, openai_token: str):
        intents = discord.Intents.all()
        intents.messages = True
        super().__init__(command_prefix="!", intents=intents)
        self.token = token
        self.openai_token = openai_token
        self.welcome_messages = ["Welcome aboard, {member_name}!",
                                 "Glad you've joined us, {member_name}!",
                                 "Hey there, {member_name}! Welcome!"]
        # Add the AppCommand cog to the bot
        self.add_cog(AppCommand(self))
        @self.command(name="airesponse", help="Get a response from OpenAI based on conversation context.")
        async def airesponse(ctx):
            model = "gpt-3.5-turbo"
            conversation_history = []
            async for message in ctx.channel.history(limit=25):
                conversation_history.append({
                    'role': 'system' if message.content.startswith('!') else ('user' if message.author == ctx.author else 'assistant'),
                    'content': message.content
                })

            response = ChatCompletion.create(
                model=model,
                messages=conversation_history,
            )
            ai_message = response['choices'][0]['message']['content']
            await ctx.reply(ai_message)

    def run(self):
        super().run(self.token)

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

        await self.add_cog(FunCommands(self))
        print("FunCommands cog loaded.")
        await self.add_cog(AdminCommands(self))
        print("AdminCommands cog loaded.")
        await self.add_cog(ResourceCommands(self))
        print("ResourceCommands cog loaded.")

    async def on_member_join(self, member: discord.Member):
        await self.greetUser(member)

    async def on_message(self, message):
        if "bot" in message.content.lower() and not message.author.bot:
            await self.spontaneous_ctx_reply(message)

        await self.process_commands(message)

    async def spontaneous_ctx_reply(self, message):
        model = "gpt-3.5-turbo"
        conversation_history = []
        async for msg in message.channel.history(limit=25):
            conversation_history.append({
                'role': 'system' if msg.content.startswith('!') else 'assistant' if msg.author == self.user else 'user',
                'content': msg.content
            })

        response = ChatCompletion.create(
            model=model,
            messages=conversation_history,
        )
        ai_message = response['choices'][0]['message']['content']
        await message.reply(ai_message)

    async def greetUser(self, member: discord.Member):
        welcome_message = random.choice(self.welcome_messages).format(member_name=member.name)
        await member.send(welcome_message)

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        print(f"Error occurred: {error}") 
        if isinstance(error, commands.errors.CommandNotFound):
            await ctx.send(f"Command not found. Use !help to see a list of available commands.")
        else:
            await ctx.send(f"There was an error with your request: {str(error)}") 

if __name__ == "__main__":
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not DISCORD_TOKEN or not OPENAI_API_KEY:
        raise ValueError("Missing required environment variables: DISCORD_TOKEN and/or OPENAI_API_KEY")

    bot = DiscordBot(DISCORD_TOKEN, OPENAI_API_KEY)
    bot.run()
