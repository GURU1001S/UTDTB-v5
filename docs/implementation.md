### Implementation: Configuration Classes
| Class | Primary Responsibility |
| :--- | :--- |
| **FlightPhase** | `IntEnum` encoding 6 flight phases (Taxi, Takeoff, Climb, Cruise, Descent, Landing). |
| **PhysicsConfig** | Stores Brayton cycle constants, fracture limits, and manufacturing variability. |
| **DegradationConfig** | Stores rate constants for fatigue, creep, corrosion, and fouling. |
| **FADECConfig** | Thresholds for EGT redlines, throttle lag, and surge margins. |
| **MaintenanceConfig** | Probabilities for compressor wash, blade repair, and sensor replacement. |
| **RareEventConfig** | Probabilities/magnitudes for bird strikes, stalls, and fuel contamination. |
| **SensorFailConfig** | Logic for stuck sensors, signal delays, quantization, and drift faults. |
| **FleetConfig** | Airline assignments, route distance ranges, and environmental exposure levels. |
| **UncertaintyConfig** | Defines aleatoric/epistemic fractions and failure probability thresholds. |
| **DatasetConfig** | Top-level settings: total engine counts, cycle ranges, and output formats. |


### Implementation: Logic & Simulation Engines
| Class | Primary Responsibility |
| :--- | :--- |
| **CausalGraph** | Generates the 19-node DAG; provides the `(19, 19)` adjacency matrix. |
| **FlightPhaseManager** | Samples representative phases per cycle based on mission duration. |
| **MissionContext** | Samples route-specific distance, duration, elevation, and sand/salt factors. |
| **TransientThermo** | Executes the ISA atmosphere model and the full Brayton cycle ODE. |
| **DegradationEngine** | Computes failure physics, Health Index (HI), and PINN residuals ($dX/dN$). |
| **FADEC** | Handles throttle dynamics, EGT protection, and bleed valve logic. |
| **MaintenanceManager** | Simulates repair actions (e.g., blade repair, module replacement). |
| **SensorLayer** | Maps physics states to 20-channel ground truth and injects 3-layer noise. |
| **TimestampGenerator** | Generates time-irregular flight epochs with simulated ACARS dropout. |
| **UncertaintyQ** | Derives RUL distributions, epistemic/aleatoric metrics, and confidence intervals. |
| **EngineSimulator** | **Orchestrator:** Simulates a single engine from birth to failure. |
| **HDF5Writer** | Handles incremental, resizable writes to the HDF5 split groups. |
| **UTDTBGenerator** | **Top-level Controller:** Manages the full fleet generation across splits. |
