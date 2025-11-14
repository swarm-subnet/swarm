import sys
from agent import RLAgent
from agent_server import start_server

if __name__ == "__main__":
    agent = RLAgent()
    start_server(agent, port=8000)

