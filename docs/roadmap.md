# Swarm Subnet — Strategic Roadmap

## Stage 0 – Baseline (“Boot-Up”)
**Goal** Deliver a reference implementation that anyone can mine or validate against in minutes.

**Objective:** Fun & Fairness  

**Description:**  
- The default code generates dynamic maps based on random seeds. Miners have to improve routes (based on successful completions, time flying and battery left) to compete.

## Stage 1 – Static Map Difficulty Ramp-Up
**Goal** Stress-test miner dones on harder, but still static environments.

**Key upgrades**  
- **Larger search space** Pole further away, elevation changes, occluding obstacles. 

## Stage 2 – Dynamic Obstacles & Moving Goals
**Goal** Introduce temporal planning.

**New mechanics**  
- NPC drones / birds with randomized trajectories.  
- Moving goal-pole (linear / circular path).  
- Collision penalties + no-fly zones.

## Stage 3 – High-Fidelity Simulation (“Simulation Jump Jump”)
**Goal** Narrow the Sim-to-Real gap.

**🚀 Migration Path:** PyBullet ➜ New Simulator (TBD)  
**🔍 Details:** Explore Gazebo, Airsim, Flighmare, Pegasus… and zero in on the ultimate real-world simulation platform!  

## Stage 4 – Long-Range Navigation (“Travel Missions”)
**Goal** Test endurance, GPS-denied odometry and multi-map stitching.

**Additions**  
- Multi-kilometre procedurally-generated landscapes.  
- Mid-point recharging pads (energy budgeting becomes critical).  
- Magnetometer & barometer sensor noise models.

## Stage 5 – Interceptor
**Goal** Intercept another drone 

- **Objective:** Your miner drone must lock onto and intercept a scripted target path **within the time budget**.  
- **Reward Terms:**  
  - ⏱️ **Capture-Time:** Faster intercepts score higher.  
  - 🛡️ **Safety:** Zero collisions—keep it clean and precise.  


## Stage 6 – Controlled Real-World Pilots
**Goal** Validate that Swarm-trained policies survive reality.

**Track 6-R (Research)**  
- Indoor motion-capture lab, 75 g micro-quads running Linux-based autopilot.  
- Automated log upload → on-chain notarisation of real-world flights.

## Stage 7 – Commercial Partnerships & Services
**Goal** Translate open research gains into sustainable value streams. We'll pitch (likely this will start around phase 5) to:

## 🚚 Last-Mile Logistics & Retail  
## 🏗 Industrial Inspection & Mapping  
## 🚒 Public Safety & Emergency Response  
## 🛡 Defense & Counter-UAS  
## 🛩 UAV Manufacturers & Autopilot Stacks  
## 🎓 Academia & Certification Bodies  


## Indicative Timeline
| Year/Q   | Stage            |
|----------|------------------|
| 2025 Q2  | 0, 1             |
| 2025 Q3  | 2                |
| 2025 Q4  | 3                |
| 2026 H1  | 4                |
| 2026 H2  | 5                |
| 2027 H1  | 6                |
| 2027 H2+ | 7                |


