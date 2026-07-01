import traci
import numpy as np
import subprocess
import sys
import os

class TrafficEnvironment:
    def __init__(self):
        self.sumo_cmd = [
            "sumo",
            "-c", "sumo_files/config.sumocfg",
            "--no-warnings",
            "--random"
        ]
        
        # 4 intersections (agents)
        self.traffic_lights = ["n1", "n2", "n3", "n4"]
        self.num_agents = len(self.traffic_lights)
        
        # 4 possible actions per agent
        # 0=Green NS, 1=Yellow NS, 2=Green EW, 3=Yellow EW
        self.num_actions = 4
        
        # state size per agent
        # [queue_length, waiting_time, density, current_phase]
        self.state_size = 4
        
        self.is_running = False

    def reset(self):
        if self.is_running:
            traci.close()
        traci.start(self.sumo_cmd)
        self.is_running = True
        self.step_count = 0
        return self._get_states()

    def _get_states(self):
        states = []
        for tl in self.traffic_lights:
            # get lanes controlled by this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(tl)
            
            # queue length
            queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                       for lane in controlled_lanes)
            
            # waiting time
            waiting = sum(traci.lane.getWaitingTime(lane) 
                         for lane in controlled_lanes)
            
            # vehicle density
            density = sum(traci.lane.getLastStepVehicleNumber(lane) 
                         for lane in controlled_lanes)
            
            # current signal phase
            phase = traci.trafficlight.getPhase(tl)
            
            # normalize values
            state = np.array([
                min(queue / 20.0, 1.0),
                min(waiting / 200.0, 1.0),
                min(density / 20.0, 1.0),
                phase / 4.0
            ], dtype=np.float32)
            
            states.append(state)
        return states

    def step(self, actions):
        # apply actions
        for i, tl in enumerate(self.traffic_lights):
            traci.trafficlight.setPhase(tl, actions[i])
        
        # advance simulation
        traci.simulationStep()
        self.step_count += 1
        
        # get new states
        next_states = self._get_states()
        
        # calculate rewards
        rewards = self._get_rewards()
        
        # check if done
        done = self.step_count >= 3600 or traci.simulation.getMinExpectedNumber() == 0
        
        return next_states, rewards, done

    def _get_rewards(self):
        rewards = []
        for tl in self.traffic_lights:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl)
            
            queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                       for lane in controlled_lanes)
            waiting = sum(traci.lane.getWaitingTime(lane) 
                         for lane in controlled_lanes)
            throughput = sum(traci.lane.getLastStepVehicleNumber(lane) 
                            for lane in controlled_lanes)
            
            # reward = penalize queue and waiting, reward throughput
            reward = -(0.4 * queue + 0.4 * waiting / 100.0) + 0.2 * throughput
            rewards.append(reward)
        return rewards

    def close(self):
        if self.is_running:
            traci.close()
            self.is_running = False