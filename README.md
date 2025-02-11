# **ORL-OPP: An Online Path Planning Framework via Offline Reinforcement Learning**

This repository contains the official implementation of **ORL-OPP**, an **Online Path Planning Framework** based on **Offline Reinforcement Learning**.

## **Introduction**

Online path planning in urban traffic systems poses significant challenges due to the need for real-time decision-making in dynamic and uncertain environments. While **Reinforcement Learning (RL)** has demonstrated promise in addressing these issues, it typically relies on **online exploration** and **simulator-based interactions**, which introduce several limitations:

- High computational costs
- Limited generalizability due to the gap between **synthetic** and **real-world** traffic data

In contrast, **Offline Reinforcement Learning (Offline RL)** utilizes large-scale historical traffic data to infer optimal routing strategies without the need for real-time exploration, making it a scalable alternative.

However, integrating **Offline RL** into online path planning comes with several key challenges:
1. **Modeling the spatial-temporal complexity** of real-world traffic dynamics.
2. **Ensuring robust generalization** to unseen traffic conditions.
3. **Efficiently utilizing historical data** for policy optimization.

To address these challenges, we propose **ORL-OPP**, an **Offline Reinforcement Learning framework for Online Path Planning**, which learns optimal routing policies from pre-collected real-world traffic data. ORL-OPP employs **Fitted Q-Iteration (FQI)** to infer optimal paths while addressing coverage deficiencies in offline datasets. Additionally, it constructs an **urban traffic graph** using **Graph Convolutional Networks (GCNs)** to model intricate spatial-temporal interactions in road networks, improving generalization to underrepresented regions. 

To further enhance **sample efficiency**, **ORL-OPP** integrates a **prioritized experience replay mechanism** that adaptively selects high-value experiences for training, ensuring more effective policy learning.

### **Key Performance**
- **16.71% reduction in travel time** compared to existing baselines, without requiring real-time or simulator feedback.

## **Data**

The dataset for ORL-OPP is available for download on [GitHub](https://github.com). Follow the steps below to get started:

1. **Download and Extract the Data**  
   Download the preprocessed data and unzip the downloaded file to your local machine.

2. **Set the Data Path**  
   In `my_constants.py`, set the `PREFIX_PATH` variable to the path where the data has been extracted.

3. **Data Structure**  
   The data is organized by city (Chengdu, Harbin, Porto, Beijing, CityIndia) and contains the following types of files:

   - **Map-Matched Pickled Trajectories**  
     A Python pickled list of tuples, where each tuple is structured as follows:
     ```python
     (trip_id, trip, time_info)
     ```
     - `trip`: A list of edge identifiers corresponding to the journey.
     - `time_info`: Contains time-related data for each trip.

   - **OSM Map Data**  
     The **map folder** contains the following files:
     - `nodes.shp`: Contains OSM node information, mapping node IDs to latitude and longitude.
     - `edges.shp`: Contains network connectivity information, mapping edge IDs to corresponding node IDs.
     - `graph_with_haversine.pkl`: A pickled **NetworkX** graph corresponding to the OSM data.

## **Installation**

To install ORL-OPP, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-repository/ORL-OPP.git
cd ORL-OPP

# Install dependencies
pip install -r requirements.txt
