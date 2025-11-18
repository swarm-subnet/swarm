from drone_agent import DroneFlightController
from agent_server import start_server

if __name__ == "__main__":
    agent = DroneFlightController()
    start_server(agent, port=8000)

