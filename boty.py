import os
import random
import discord
import psycopg2
import openai
import aiohttp
import json
from discord.ext import commands
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from openai import ChatCompletion
import asyncio
from langchain.schema import SystemMessage
import logging


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

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
                dbname=os.getenv("POSTGRES_DB", "mydatabase"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host=os.getenv("POSTGRES_HOST", "localhost")
            )
        except psycopg2.OperationalError as e:
            logging.error(f"Could not connect to PostgreSQL database: {e}")
            self.conn = None

    @commands.command(help="Send a resource to a user.")
    async def sendResource(self, ctx, member: discord.Member):
        resource_channel = discord.utils.get(ctx.guild.channels, name='resources')
        resource_messages = [m async for m in resource_channel.history(limit=10)]
    
        resource_links = [m.content for m in resource_messages if m.content.startswith('http')]
    
        with self.conn.cursor() as cur:
            for link in resource_links:
                cur.execute("INSERT INTO resources (link) VALUES (%s) ON CONFLICT (link) DO NOTHING", (link,))
        self.conn.commit()

        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:5001/get_resource_data') as response:
                data = await response.json()

        for resource in data.values():
            report = f"Title: {resource['title']}\nLink: {resource['link']}\nDescription: {resource['description']}"
        await member.send(report)

class ResearchCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.agent = self.initialize_agent()
        logging.info("ResearchCommands initialized")

    def initialize_agent(self):
        # Move the agent initialization code here
        # Create the agent
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
        memory = ConversationSummaryBufferMemory(
            memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )
        return agent

    @commands.command(name='research')
    async def research(self, ctx, *, query):
        logging.info("Research command invoked")
        print("Query: ", query)
        result = self.agent({"input": query})
        await ctx.send(result['output'])

class MyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=discord.Intents.all())
        self.add_cog(FunCommands(self))
        self.add_cog(AdminCommands(self))
        self.add_cog(ResourceCommands(self))
        self.add_cog(ResearchCommand(self))  # Fixed the class name here
        logging.info("MyBot initialized")


# Tool for search
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text

# Tool for scraping
def scrape_website(objective: str, url: str):
    headers = {'Cache-Control': 'no-cache', 'Content-Type': 'application/json'}
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# Summary function
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

# ScrapeWebsiteTool class
class ScrapeWebsiteInput(BaseModel):
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput
    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    def _arun(self, url: str):
        raise NotImplementedError("error: not implemented")

# Create langchain agent with the tools above
tools = [
    Tool(name="Search", func=search, description="useful for when you need to answer questions about current events, data. You should ask targeted questions"),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(content="""
You are a world class researcher, who can do detailed research on any topic and produce facts based results; you do not make things up, you will try as hard as possible to gather facts & data to back up the research
Please make sure you complete the objective above with the following rules:
1/ You should do enough research to gather as much information as possible about the objective
2/ If there are relevant links & articles, scrape them to gather more information.
3/ After scraping and searching, think if there's any new things you should search and scrape based on the data you collected to increase research quality. Only do this for a maximum of 3 iterations.
4/ Only write facts and data that you have gathered, do not make things up.
5/ Include all reference data and links to back up your research in the final output.
""")

# Define placeholder
placeholder = MessagesPlaceholder(variable_name='memory')

# Define agent_kwargs
agent_kwargs = {
    "extra_prompt_messages": [placeholder],
    "system_message": system_message,
}

class MainBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=discord.Intents.all())
        self.add_cog(FunCommands(self))
        self.add_cog(AdminCommands(self))
        self.add_cog(ResourceCommands(self))
        self.add_cog(ResearchCommands(self))
        logging.info("MainBot initialized")

    async def on_ready(self):
        logging.info(f"Logged in as {self.user.name} (ID: {self.user.id})")

if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("Missing required environment variable: DISCORD_TOKEN")

    bot = MainBot()
    bot.run(TOKEN)