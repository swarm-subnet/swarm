from drone_agent import DroneFlightController
from agent_server import start_server
import os

if __name__ == "__main__":
    agent = DroneFlightController()
    port = int(os.getenv("RPC_PORT", "9000"))
    auth_token = os.getenv("AUTH_TOKEN")
    start_server(agent, port=port, auth_token=auth_token)

