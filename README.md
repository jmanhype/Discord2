# Discord2

A Discord bot with AI-powered research capabilities, integrating OpenAI, LangChain, and web scraping tools.

## Features

- **AI Research Agent**: Perform detailed research on any topic using GPT-3.5-turbo
- **Discord Bot Commands**:
  - Fun commands (ping, guess)
  - Admin commands (warn, announce, resolve)
  - Resource management
  - Research capabilities integrated into Discord
- **Web Scraping**: Automated web content extraction and summarization
- **Streamlit & FastAPI**: Multiple deployment options

## Prerequisites

- Python 3.8+
- Discord Bot Token
- OpenAI API Key
- Serper API Key (for search functionality)
- Browserless API Key (for web scraping)
- PostgreSQL database (optional, for resource management)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Discord2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```bash
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
SERP_API_KEY=your_serper_api_key
BROWSERLESS_API_KEY=your_browserless_api_key
POSTGRES_USER=your_postgres_username
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_database_name
POSTGRES_HOST=localhost
```

## Usage

### Running the Discord Bot

**Option 1: Basic bot (bot.py)**
```bash
python bot.py
```

**Option 2: Advanced bot with research capabilities (boty.py)**
```bash
python boty.py
```

### Running the Streamlit App

```bash
streamlit run app.py
```

### Running the FastAPI Server

```bash
uvicorn app:app --reload --port 8000
```

## Discord Bot Commands

- `!ping` - Check if bot is responsive
- `!guess <number>` - Guess a random number between 1-10
- `!warn @user <reason>` - Warn a user (Admin)
- `!announce <message>` - Announce to #general channel (Admin)
- `!resolve <issue>` - Mark an issue as resolved (Admin)
- `!research <query>` - Perform AI-powered research on a topic
- `!sendResource @user` - Send curated resources to a user

## Architecture

- **bot.py**: Basic Discord bot with commands and OpenAI integration
- **boty.py**: Advanced bot with LangChain research agent
- **app.py**: Streamlit UI and FastAPI endpoint for research agent
- **pp.py**: Simple Streamlit demo

## Security Notes

⚠️ **IMPORTANT**: Never commit API keys or tokens to version control. Always use environment variables via `.env` file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.