# Swarm Subnet — Strategic Roadmap

## Stage 0 – Baseline (“Boot-Up”)
**Goal** Deliver a reference implementation that anyone can mine or validate against in minutes.

**Objective:** Fun & Fairness  

**Description:**  
- The default code generates dynamic maps based on random seeds. Miners have to improve routes (based on successful completions, time flying and battery left) to compete.

## Stage 1 – Static Map Difficulty Ramp-Up
**Goal** Stress-test miner drones on harder, but still static environments.

**Key upgrades**  
- **Larger search space** Pole further away, elevation changes, occluding obstacles. 

## Stage 2 – Dynamic Obstacles & Moving Goals
**Goal** Introduce temporal planning.

**New mechanics**  
- NPC drones / birds with randomized trajectories.  
- Moving goal-pole (linear / circular path).  
- Collision penalties + no-fly zones.

## Stage 3 – Real world testing
**Goal** To validate Swarm Miner's algorithms on real hardware, we will run trials with physical drones in realistic settings. The roadmap is:

- **Sensor‑network adaptation:** Tune the SN so it meets all operational requirements for autonomous flight on actual drones.
- **Initial proof‑of‑concept flights:** Begin with straightforward indoor missions in a controlled, well‑instrumented test environment.

## Stage 4 – High-Fidelity Simulation ("Simulation Jump")
**Goal** Narrow the Sim-to-Real gap.

**🚀 Migration Path:** PyBullet ➜ New Simulator (TBD)  
**🔍 Details:** Explore Gazebo, Airsim, Flightmare , Pegasus… and zero in on the ultimate real-world simulation platform!  

## Stage 5 – Generic drone flight ("Random Drone")
**Goal** Enable miners to competently operate a variety of commercially available drones—each with its own weight, power output, battery capacity, and flight dynamics.

**Additions**  
- A training fleet that includes multiple real-world drone models, covering a broad spectrum of sizes and performance characteristics
- At this point we will start to talk to real business to adapt the SN output to their needs  

## Stage 6 – Long-Range Navigation ("Travel Missions")
**Goal** Test endurance, GPS-denied odometry and multi-map stitching.

**Additions**  
- Multi-kilometre procedurally-generated landscapes.  
- Mid-point recharging pads (energy budgeting becomes critical).  
- Magnetometer & barometer sensor noise models.

## Stage 7 – Interceptor
**Goal** Intercept another drone 

- **Objective:** Your miner drone must lock onto and intercept a scripted target path **within the time budget**.  
- **Reward Terms:**  
  - ⏱️ **Capture-Time:** Faster intercepts score higher.  
  - 🛡️ **Safety:** Zero collisions—keep it clean and precise.  


## Stage 8 – Commercial Partnerships & Services
**Goal** Translate open research gains into sustainable value streams. We'll pitch (likely this will start around phase 5) to:

- 🚚 Last-Mile Logistics & Retail  
- 🏗 Industrial Inspection & Mapping  
- 🚒 Public Safety & Emergency Response  
- 🛡 Defense & Counter-UAS  
- 🛩 UAV Manufacturers & Autopilot Stacks  
- 🎓 Academia & Certification Bodies  


## Indicative Timeline
| Year/Q   | Stage            |
|----------|------------------|
| 2025 Q2  | 0, 1             |
| 2025 Q3  | 2                |
| 2025 Q4  | 3                |
| 2026 H1  | 4                |
| 2026 H2  | 5                |
| 2027 H1  | 6                |
| 2027 H2+ | 7, 8                |

