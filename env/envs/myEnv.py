import gym
import heapq
import random
import networkx as nx
from itertools import islice
from networkx.algorithms.coloring.greedy_coloring import strategy_saturation_largest_first
import numpy as np
from service import Service
import copy
import matplotlib.pyplot as plt

import service


class MyEnv(gym.Env):

    def __init__(self, episode_length, load,
                 mean_service_holding_time, k_paths, topology_num=None):
        self.current_time = 0
        self.allocated_Service = []
        self.episode_length = episode_length
        self.services_processed = 0
        self.services_accepted = 0
        self.services_node_failed = 0
        self.services_path_failed = 0
        self.accepted = False
        self.node_accepted = False
        self.path_accepted = False
        self.reason = np.array([0]*4)


        self.load = load
        self.mean_service_holding_time = mean_service_holding_time
        self.mean_service_inter_arrival_time = 0

        self.k_paths = k_paths

        self.rng = random.Random(41)

        self.set_load(load, mean_service_holding_time)

        self.topology_num = topology_num
        # create topology of substrate network
        # 5-node topology
        if topology_num is None:
            self.topology_num = 0
            self.topology = nx.Graph()
            self.num_nodes = 5
            self.num_slots = 8
            self.num_links = 6
            for i in range(5):
                self.topology.add_node(i, capacity=5)

            for j in range(4):
                self.topology.add_edge(j, j + 1, slots=np.ones(self.num_slots, dtype=int))

            self.topology.add_edge(0, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 4, slots=np.ones(self.num_slots, dtype=int))

            # action_space
            self.action_space = gym.spaces.MultiDiscrete((self.num_nodes ** 3, self.k_paths ** 3, self.num_slots ** 3))

            # observation_space
            self.a = gym.spaces.MultiDiscrete((2 ** 3, 3 ** 3, 6 ** self.num_nodes))
            self.b = gym.spaces.Box(low=0, high=1, shape=(self.num_slots * self.num_links,), dtype=int)
            self.observation_space = gym.spaces.Dict(
                {'Vcap_Vbw_Scap': self.a,
                 'slots': self.b}
            )

            # create initialized virtual network observation
            self.current_VN_capacity = np.zeros(3, dtype=int)
            self.current_VN_bandwidth = np.zeros(3, dtype=int)

        # 5-node topology
        elif topology_num == 1:
            self.topology = nx.Graph()
            self.num_nodes = 5
            self.num_slots = 8
            self.num_links = 6

            for i in range(5):
                self.topology.add_node(i, capacity=5)

            self.topology.add_edge(0, 1, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(0, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 3, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 4, slots=np.ones(self.num_slots, dtype=int))

            # action_space
            self.action_space = gym.spaces.MultiDiscrete((self.num_nodes ** 3, self.k_paths ** 3, self.num_slots ** 3))

            # observation_space
            self.a = gym.spaces.MultiDiscrete((2 ** 3, 3 ** 3, 6 ** self.num_nodes))
            self.b = gym.spaces.Box(low=0, high=1, shape=(self.num_slots * self.num_links,), dtype=int)
            self.observation_space = gym.spaces.Dict(
                {'Vcap_Vbw_Scap': self.a,
                 'slots': self.b}
            )

            # create initialized virtual network observation
            self.current_VN_capacity = np.zeros(3, dtype=int)
            self.current_VN_bandwidth = np.zeros(3, dtype=int)


        # 9-node topology
        elif topology_num == 2:
            self.topology = nx.Graph()
            self.num_nodes = 9
            self.num_slots = 16
            self.num_links = 13

            # self.topology.add_edge(0, 1, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(0, 2, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(0, 3, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 2, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 3, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 5, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 6, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 7, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(4, 5, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(4, 8, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(5, 6, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(6, 8, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(7, 8, slots=np.ones(self.num_slots, dtype=int))

            # action_space
            self.action_space = gym.spaces.MultiDiscrete((self.num_nodes ** 3, self.k_paths ** 3, self.num_slots ** 3))

            # observation_space
            self.a = gym.spaces.MultiDiscrete((2 ** 3, 3 ** 3, 6 ** self.num_nodes))
            self.b = gym.spaces.Box(low=0, high=1, shape=(self.num_slots * self.num_links,), dtype=int)
            self.observation_space = gym.spaces.Dict(
                {'Vcap_Vbw_Scap': self.a,
                 'slots': self.b}
            )

            # create initialized virtual network observation
            self.current_VN_capacity = np.zeros(3, dtype=int)
            self.current_VN_bandwidth = np.zeros(3, dtype=int)

        # 9-node topology
        elif topology_num == 3:
            self.topology = nx.Graph()
            self.num_nodes = 9
            self.num_slots = 16
            self.num_links = 13

            # self.topology.add_edge(0, 1, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(0, 1, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(0, 2, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 2, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 3, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 5, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(4, 8, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(5, 6, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(5, 7, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(5, 8, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(6, 7, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(7, 8, slots=np.ones(self.num_slots, dtype=int))

            # action_space
            self.action_space = gym.spaces.MultiDiscrete((self.num_nodes ** 3, self.k_paths ** 3, self.num_slots ** 3))

            # observation_space
            self.a = gym.spaces.MultiDiscrete((2 ** 3, 3 ** 3, 6 ** self.num_nodes))
            self.b = gym.spaces.Box(low=0, high=1, shape=(self.num_slots * self.num_links,), dtype=int)
            self.observation_space = gym.spaces.Dict(
                {'Vcap_Vbw_Scap': self.a,
                 'slots': self.b}
            )

            # create initialized virtual network observation
            self.current_VN_capacity = np.zeros(3, dtype=int)
            self.current_VN_bandwidth = np.zeros(3, dtype=int)
       
        # 5-node topology
        elif topology_num == 4:
            self.topology = nx.Graph()
            self.num_nodes = 5
            self.num_slots = 8
            self.num_links = 7

            for i in range(5):
                self.topology.add_node(i, capacity=5)

            for j in range(4):
                self.topology.add_edge(j, j + 1, slots=np.ones(self.num_slots, dtype=int))

            self.topology.add_edge(0, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 3, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 4, slots=np.ones(self.num_slots, dtype=int))
            
            # action_space
            self.action_space = gym.spaces.MultiDiscrete((self.num_nodes ** 3, self.k_paths ** 3, self.num_slots ** 3))

            # observation_space
            self.a = gym.spaces.MultiDiscrete((2 ** 3, 3 ** 3, 6 ** self.num_nodes))
            self.b = gym.spaces.Box(low=0, high=1, shape=(self.num_slots * self.num_links,), dtype=int)
            self.observation_space = gym.spaces.Dict(
                {'Vcap_Vbw_Scap': self.a,
                 'slots': self.b}
            )

            # create initialized virtual network observation
            self.current_VN_capacity = np.zeros(3, dtype=int)
            self.current_VN_bandwidth = np.zeros(3, dtype=int)


        # 8-node topology
        elif topology_num == 5:
            self.topology = nx.Graph()
            self.num_nodes = 8
            self.num_slots = 16
            self.num_links = 13

            self.topology.add_edge(0, 1, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(0, 2, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 2, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(1, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 3, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(2, 5, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 4, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 5, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(3, 6, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(4, 7, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(5, 6, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(5, 7, slots=np.ones(self.num_slots, dtype=int))
            self.topology.add_edge(6, 7, slots=np.ones(self.num_slots, dtype=int))

            # action_space
            self.action_space = gym.spaces.MultiDiscrete((self.num_nodes ** 3, self.k_paths ** 3, self.num_slots ** 3))

            # observation_space
            self.a = gym.spaces.MultiDiscrete((2 ** 3, 3 ** 3, 6 ** self.num_nodes))
            self.b = gym.spaces.Box(low=0, high=1, shape=(self.num_slots * self.num_links,), dtype=int)
            self.observation_space = gym.spaces.Dict(
                {'Vcap_Vbw_Scap': self.a,
                 'slots': self.b}
            )

            # create initialized virtual network observation
            self.current_VN_capacity = np.zeros(3, dtype=int)
            self.current_VN_bandwidth = np.zeros(3, dtype=int)

        


    def step(self, action):
        print('topology', self.topology_num)
        print('time step ', self.services_processed)
        print('current VN C ', self.current_VN_capacity)
        print('current VN BW ', self.current_VN_bandwidth)
        # step function has two part. In Part1, it checks whether the action meets the constraints,
        # In Part2, it pushes the time to the moment just before the new request arriving, and send it to the agent
        #
        ######################################################################################################
        # virtual network contains 3 nodes and 3 links in this case.
        # The observation and action spaces choose from the number of combinations of nodes, k_path, initial slots,
        # the combination could be treated as integer number in different bases.
        # For example, 3 virtual nodes (VN0, VN1, VN2) are mapping to 5 substrate nodes (SN0, SN1, SN2, SN3, SN4),
        #       VN0 might be mapped to any one of 5 substrate nodes, and same for VN1 and VN2,
        #       thus, the mapping action could be represented by integer within the interval [000,444],
        #       which is a Hexadecimal (base 5) number.
        #       The number of integers within the interval is just the action space.
        #       If we do base conversion, we can transfer Decimal number to Hexadecimal number,
        #       and hence know the nodes.
        # Same for k_path action and initial slot action

        # nodes selected from the action
        node2 = int(action[0] / self.num_nodes ** 2)
        node1 = int((action[0] - node2 * self.num_nodes ** 2) / self.num_nodes ** 1)
        node0 = int((action[0] - node2 * self.num_nodes ** 2 - node1 * self.num_nodes ** 1) / self.num_nodes ** 0)
        nodes_selected = np.array([node0, node1, node2], dtype=int)
        print('nodes_selected: ')
        print(nodes_selected)

        # k_paths selected from the action
        # k0 is the link between node0 and node1
        # k1 is the link between node0 and node2
        # k2 is the link between node1 and node2
        k2 = int(action[1] / self.k_paths ** 2)
        k1 = int((action[1] - k2 * self.k_paths ** 2) / self.k_paths ** 1)
        k0 = int((action[1] - k2 * self.k_paths ** 2 - k1 * self.k_paths ** 1) / self.k_paths ** 0)
        k_path_selected = np.array([k0, k1, k2], dtype=int)
        print('k_path_selected: ')
        print(k_path_selected)

        # initial slot selected from the action
        s2 = int(action[2] / self.num_slots ** 2)
        s1 = int((action[2] - s2 * self.num_slots ** 2) / self.num_slots ** 1)
        s0 = int((action[2] - s2 * self.num_slots ** 2 - s1 * self.num_slots ** 1) / self.num_slots ** 0)
        initial_slot_selected = np.array([s0, s1, s2], dtype=int)
        print('initial_slot_selected: ')
        print(initial_slot_selected)

        # make sure different substrate nodes are used & check substrate node capacity
        node_free = True
        if (nodes_selected[0] != nodes_selected[1]) & (nodes_selected[0] != nodes_selected[2]) \
                & (nodes_selected[1] != nodes_selected[2]):
            for i in range(3):
                node_free = node_free & self.is_node_free(nodes_selected[i], self.current_VN_capacity[i])
        else:
            node_free = False

        # make sure different path slots are used & check substrate link BW
        path_free = True
        path_list = [0, 0, 0]

        self.num_path_accepted = 0
        if node_free is True:
            # create matrices that show the connection of a path. they are symmetric matrices
            path_matrix = np.zeros((3, self.num_nodes, self.num_nodes), dtype=int)
            # connection gives the source and destination of a virtual link
            connection = np.array([[0, 1],
                                   [0, 2],
                                   [1, 2]], dtype=int)
            for i in range(3):
                source = nodes_selected[connection[i, 0]]
                destination = nodes_selected[connection[i, 1]]
                if k_path_selected[i] + 1 > len(self.get_k_shortest_paths(self.topology, source, destination,
                                                                          k_path_selected[i] + 1)):
                    path_free = False
                else:
                    path = \
                        self.get_k_shortest_paths(self.topology, source, destination, k_path_selected[i] + 1)[
                            k_path_selected[i]]
                    print(path)
                    path_list[i] = path
                    for j in range(len(path) - 1):
                        path_matrix[i, path[j], path[j + 1]] = 1
                        path_matrix[i, path[j + 1], path[j]] = 1

                    path_free = path_free & self.is_path_free(path, initial_slot_selected[i],
                                                              self.current_VN_bandwidth[i])
                    if self.is_path_free(path, initial_slot_selected[i], self.current_VN_bandwidth[i]):
                        self.num_path_accepted += 1

            # we need to make sure the slots in the substrate network do not crash each other
            # use the path_matrices created in the previous step to check if there are links that be used many times
            # if such a link exists, we check the slot usage of the two path
            # check if the initial slot with larger index lies within the occupied slots of another path.
            for k in range(3):
                path_matrix_check = path_matrix[connection[k, 0], :, :] & path_matrix[connection[k, 1], :, :] == 0
                if not np.all(path_matrix_check == True):
                    min_is = min(initial_slot_selected[connection[k, 0]], initial_slot_selected[connection[k, 1]])
                    max_is = max(initial_slot_selected[connection[k, 0]], initial_slot_selected[connection[k, 1]])
                    min_is_bw = 0
                    if min_is == initial_slot_selected[connection[k, 0]]:
                        min_is_bw = self.current_VN_bandwidth[connection[k, 0]]
                    else:
                        min_is_bw = self.current_VN_bandwidth[connection[k, 1]]

                    if min_is + min_is_bw - 1 >= max_is:
                        path_free = False

        # accepted?
        self.accepted = node_free & path_free      
        self.node_accepted = node_free

        self.path_accepted = path_free
        if self.node_accepted is False:
          self.services_node_failed += 1
        if self.path_accepted is False:
          self.services_path_failed += 1

        # print('accepted? ', self.accepted)
        if self.accepted:
            ht = self.rng.expovariate(1 / self.mean_service_holding_time)
            current_service = Service(copy.deepcopy(self.current_time), ht,
                                      nodes_selected, copy.deepcopy(self.current_VN_capacity),
                                      path_list, copy.deepcopy(self.current_VN_bandwidth),
                                      initial_slot_selected)
            self.add_to_list(current_service)
            self.map_service(current_service)
            self.services_accepted += 1

        # self.print_topology()

        self.set_load(self.load, self.mean_service_holding_time)
        self.traffic_generator()
        self.request_generator()
        reward = self.reward()
        print('reward', reward)
        if reward == 10:
          self.reason[0] += 1          
        if reward == -5:
          self.reason[1]+=1
        if reward == -8:
          self.reason[2]+=1
        if reward ==-10:
          self.reason[3]+=1
        print('reason',self.reason)
        
      
            
       

        self.services_processed += 1

        observation = self.observation()
        done = self.services_processed == self.episode_length
        info = {'P_accepted': self.services_accepted / self.services_processed,
                'topology_num': self.topology_num,
                'load': self.load,
                'reason': self.reason,
                'NF': self.services_node_failed / self.services_processed,
                'MF': self.services_path_failed / self.services_processed,
                'SF': (self.services_node_failed+self.services_path_failed)/self.services_processed,
                'BP': 1-self.services_accepted / self.services_processed,
                'D':(1-self.services_accepted / self.services_processed)-((self.services_node_failed+self.services_path_failed)/self.services_processed)}

        # print substrate network at the moment before mapping the new virtual network
        # self.print_topology()

        return observation, reward, done, info

    def reset(self):
        self.current_time = 0
        self.allocated_Service = []
        self.services_processed = 0
        self.services_accepted = 0

        self.services_node_failed = 0
        self.services_path_failed = 0
        self.path_accepted = False
        self.node_accepted = False

        self.accepted = False

        for i in range(np.array(self.topology.nodes).shape[0]):
            self.topology.nodes[i]['capacity'] = 5

        for j in range(np.array(self.topology.edges).shape[0]):
            self.topology.edges[np.array(self.topology.edges)[j]]['slots'] = np.ones(self.num_slots, dtype=int)

        self.request_generator()
        observation = self.observation()

        return observation

    def render(self, mode="human"):
        return self.topology, self.num_slots


    def reward0(self):
        """if self.accepted:
            reward = 10
        else:
            reward = 0"""

        if self.accepted:
            reward = 20
            
        else:
            if self.node_accepted:
                if self.num_path_accepted > 1:
                    reward = 10
                else:
                    reward = -10
            else:
                reward = -20

        return reward


    def reward(self):
        """if self.accepted:
            reward = 10
        else:
            reward = -10"""

        if self.accepted:
            reward = 10
            
        else:
            if self.node_accepted:
                if self.num_path_accepted > 1:
                    reward = -5
                    
                else:
                    reward = -8
                    
            else:
                reward = -10
                

        return reward



    def get_k_shortest_paths(self, g, source, target, k, weight=None):
        """
        Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
        """
        return list(islice(nx.shortest_simple_paths(g, source, target, weight=weight), k))

    def is_node_free(self, n, capacity_required):
        n_capacity = self.topology.nodes[n]['capacity']
        if n_capacity < capacity_required:
            return False
        else:
            return True

    def is_path_free(self, path, initial_slot, num_slots):
        if initial_slot + num_slots > self.num_slots:
            return False

        path_slots = np.ones(self.num_slots, dtype=int)
        for i in range(len(path) - 1):
            path_slots = path_slots & self.topology.edges[path[i], path[i + 1]]['slots']

        if np.sum(path_slots[initial_slot:initial_slot + num_slots]) < num_slots:
            return False
        return True

    def add_to_list(self, service: Service):
        heapq.heappush(self.allocated_Service, (service.arrival_time + service.holding_time, service))

    def map_service(self, service: Service):
        # nodes mapping
        for i in range(len(service.nodes)):
            node_num = service.nodes[i]
            self.topology.nodes[node_num]['capacity'] -= service.nodes_capacity[i]

        # links mapping
        for i in range(len(service.path)):
            slots_occupied = np.zeros(self.num_slots, dtype=int)
            for j in range(service.links_BW[i]):
                slots_occupied[service.links_IS[i] + j] = 1

            for k in range(len(service.path[i]) - 1):
                s = service.path[i][k]
                d = service.path[i][k + 1]

                self.topology.edges[s, d]['slots'] -= slots_occupied

    def set_load(self, load, mean_service_holding_time):
        self.mean_service_inter_arrival_time = 1 / float(load / float(mean_service_holding_time))

    def release_service(self, service: Service):
        # nodes release
        for i in range(len(service.nodes)):
            node_num = service.nodes[i]
            self.topology.nodes[node_num]['capacity'] += service.nodes_capacity[i]

        # links release
        for i in range(len(service.path)):
            slots_occupied = np.zeros(self.num_slots, dtype=int)
            for j in range(service.links_BW[i]):
                slots_occupied[service.links_IS[i] + j] = 1

            for k in range(len(service.path[i]) - 1):
                s = service.path[i][k]
                d = service.path[i][k + 1]
                self.topology.edges[s, d]['slots'] += slots_occupied

    def traffic_generator(self):
        """
        Method from https://github.com/carlosnatalino/optical-rl-gym/blob/fc9a82244602d8efab749fe4391c7ddb4b05dfe7/optical_rl_gym/envs/rmsa_env.py#L280
        """
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        while len(self.allocated_Service) > 0:
            (time, service_to_release) = heapq.heappop(self.allocated_Service)
            if time <= self.current_time:
                self.release_service(service_to_release)
            else:
                heapq.heappush(self.allocated_Service, (service_to_release.arrival_time
                                                        + service_to_release.holding_time, service_to_release))
                break

    def request_generator(self):
        for i in range(len(self.current_VN_capacity)):
            self.current_VN_capacity[i] = self.rng.randint(1, 2)

        for i in range(len(self.current_VN_bandwidth)):
            self.current_VN_bandwidth[i] = self.rng.randint(2, 4)

    def observation(self):
        # VN observation + SN observation
        obs_dict = {'Vcap_Vbw_Scap': np.zeros(3, dtype=int),
                    'slots': np.zeros(self.num_slots * self.num_links, dtype=int)}
        Vcap_array = self.current_VN_capacity - 1
        Vcap_int = 0
        for i in range(len(Vcap_array)):
            Vcap_int += Vcap_array[i] * 2 ** i

        Vbw_array = self.current_VN_bandwidth - 2
        Vbw_int = 0
        for i in range(len(Vbw_array)):
            Vbw_int += Vbw_array[i] * 3 ** i

        Scap_array = np.zeros(self.num_nodes, dtype=int)
        Scap_int = 0
        for i in range(self.num_nodes):
            Scap_array[i] = self.topology.nodes[i]['capacity']

        for j in range(len(Scap_array)):
            Scap_int += Scap_array[j] * 6 ** j

        obs_dict['Vcap_Vbw_Scap'] = [Vcap_int, Vbw_int, Scap_int]

        Sslots_matrix = np.zeros((self.num_links, self.num_slots), dtype=int)
        for i in range(self.num_links):
            s_d = np.array(self.topology.edges)[i]
            Sslots_matrix[i, :] = self.topology.edges[s_d]['slots']


        obs_dict['slots'] = Sslots_matrix.reshape(self.b.shape)
        return obs_dict

    def print_topology(self):
        SN_C = np.zeros(self.num_nodes, dtype=int)
        SN_slots = np.zeros((self.num_links, self.num_slots), dtype=int)
        for i in range(len(self.topology.nodes)):
            SN_C[i] = self.topology.nodes[i]['capacity']

        for i in range(self.num_links):
            SN_slots[i, :] = self.topology.edges[np.array(self.topology.edges)[i]]['slots']
        print('SN_C ', SN_C)
        print('SN_slots ', SN_slots)
        print('number of services', len(self.allocated_Service))


