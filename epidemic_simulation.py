import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time
import os



class EpidemicSimulation:
    def __init__(
        self,
        infection_rate,
        recover_rate,
        num_people,
        initial_infected,
        num_clusters,
        intra_connections_per_cluster,
        total_inter_connections,
        num_steps=100,
        measure_interval=5,
        method="quantum",  # 可選 "quantum" 或 "classical"
        network_type="random",
        save_csv=True,
        csv_filename="epidemic_results.csv",
        exp_name=""
    ):
        """
        初始化傳染病模擬參數
        """
        self.infection_rate = infection_rate
        self.recover_rate = recover_rate
        self.num_people = num_people
        self.initial_infected = initial_infected
        self.num_clusters = num_clusters
        self.intra_connections_per_cluster = intra_connections_per_cluster
        self.total_inter_connections = total_inter_connections
        self.num_steps = num_steps
        self.measure_interval = measure_interval
        self.method = method.lower()  # "quantum" or "classical"
        self.save_csv = save_csv
        self.csv_filename = csv_filename
        self.network_type = network_type
        # 設定實驗的唯一識別碼（基於時間戳記）
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"epidemic_{timestamp}_I{infection_rate}_R{recover_rate}_P{num_people}_C{num_clusters}_S{num_steps}.csv"
        self.results_dir = f"results/{self.method}/diff_{exp_name}_{network_type}/I{infection_rate}_R{recover_rate}_P{num_people}_C{num_clusters}_S{num_steps}_intra{intra_connections_per_cluster}_inter{total_inter_connections}_measure_interval{measure_interval}/"

        # 確保資料夾存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.global_state = [0] * num_people
        for i in random.sample(range(num_people), initial_infected):
            self.global_state[i] = 1
        self.infection_counts = []

        self._initialize_network(self.network_type)
        if self.method == "quantum":
            self.simulator = AerSimulator()

    def _inverse_sin_squared(self, y):
        """計算等效的傳染率與恢復率 (僅量子方法使用)"""
        if y < 0 or y > 1:
            raise ValueError("Input y must be in the range [0, 1]")
        return (2 / np.pi) * np.arcsin(np.sqrt(y))

    def _initialize_network(self, network_type="random"):
        """初始化網絡結構，支援不同類型的網絡拓撲"""
        
        self.clusters = [list(range(i * (self.num_people // self.num_clusters),
                                    (i + 1) * (self.num_people // self.num_clusters)))
                        for i in range(self.num_clusters)]

        self.intra_connections = {i: set() for i in range(self.num_clusters)}
        self.inter_connections = set()
        # 量子模式計算等效傳染率
        if self.method == "quantum":
            self.infect = self._inverse_sin_squared(self.infection_rate)
            self.recover = self._inverse_sin_squared(self.recover_rate)
        # **選擇網絡結構**
        if network_type == "random":
            self._initialize_random_network()
        elif network_type == "scale-free":
            self._initialize_scale_free_network()
        elif network_type == "small-world":
            self._initialize_small_world_network()
        else:
            raise ValueError("❌ Invalid network type. Choose from: 'random', 'scale-free', 'small-world'.")

    def _initialize_random_network(self):
        """使用隨機連線方式（與原本相同）"""
        for i in range(self.num_clusters):
            while len(self.intra_connections[i]) < self.intra_connections_per_cluster:
                src, tgt = random.sample(self.clusters[i], 2)
                self.intra_connections[i].add((src, tgt))

        while len(self.inter_connections) < self.total_inter_connections:
            src_cluster, tgt_cluster = random.sample(range(self.num_clusters), 2)  # 確保不同區塊
            src = random.choice(self.clusters[src_cluster])
            tgt = random.choice(self.clusters[tgt_cluster])
            self.inter_connections.add((src, tgt))

        # 計算總連線數
        self.quantum_connection_count = sum(len(conns) for conns in self.intra_connections.values())
        self.classical_connection_count = len(self.inter_connections)
        
        # print(f"Quantum Connections: {self.quantum_connection_count}")
        # print(f"Classical Connections: {self.classical_connection_count}")
        # print('random')

    def _initialize_small_world_network(self, p=0.1):
        """
        Small-world (position-aligned) initialization that:
        - For each cluster pair (i -> i+1), first creates edges using offset rounds:
            offset = 0 => node n -> node n
            offset = 1 => node n -> node n+1
            ...
            until the required count for that pair is reached.
        - If the required count exceeds number of unique pairs, continue offset rounds
            but skip already-added pairs (so no duplicates).
        - After building all pair-edges, perform rewiring on a fraction p of total edges.
        """
        import random
        from collections import defaultdict

        self.inter_connections = set()
        self.intra_connections = defaultdict(set)

        # Step 1: maintain original intra-cluster random connections
        for i in range(self.num_clusters):
            while len(self.intra_connections[i]) < self.intra_connections_per_cluster:
                src, tgt = random.sample(self.clusters[i], 2)
                self.intra_connections[i].add((src, tgt))

        # Step 2: cluster ring
        cluster_ring = [(i, (i + 1) % self.num_clusters) for i in range(self.num_clusters)]
        m = len(cluster_ring)

        # distribute total_inter_connections among ring pairs (evenly, remainder to first pairs)
        base_count = self.total_inter_connections // m
        remainder = self.total_inter_connections % m
        per_pair_counts = [base_count + (1 if idx < remainder else 0) for idx in range(m)]

        all_edges = []  # list to collect edges in generation order

        # generate edges per pair using offset-round rule (offset=0 -> n->n)
        for pair_idx, (i, j) in enumerate(cluster_ring):
            need = per_pair_counts[pair_idx]
            if need <= 0:
                continue

            cluster_i = self.clusters[i]
            cluster_j = self.clusters[j]
            len_i = len(cluster_i)
            len_j = len(cluster_j)

            # set to track which pairs already added for this pair (to avoid duplicates)
            added = set()

            offset = 0
            # keep looping offsets until we've added 'need' unique edges or we've exhausted possible unique pairs
            while len(added) < need:
                # if we've already explored all possible offsets and added all unique pairs, break to avoid infinite loop
                if offset > max(len_j, len_i) * 2 and len(added) >= len_i * len_j:
                    # all unique pairs added
                    break

                for src_idx in range(len_i):
                    if len(added) >= need:
                        break
                    tgt_idx = (src_idx + offset) % len_j
                    u = cluster_i[src_idx]
                    v = cluster_j[tgt_idx]
                    if (u, v) not in added:
                        added.add((u, v))
                        all_edges.append((u, v))
                        # continue until we've reached 'need'

                offset += 1

                # safety: if we've added all possible unique pairs already and still need more (shouldn't happen unless need>len_i*len_j)
                if len(added) >= len_i * len_j and len(added) < need:
                    # we have exhausted unique pairs; at this point we've followed your rule adding in offset order
                    # If you want to allow duplicates beyond this point, uncomment the block below to allow repeating in same offset order.
                    # For now, we'll stop adding unique pairs (so actual edges per pair == len_i*len_j).
                    break

        # Step 3: rewiring p fraction of all_edges
        total_edges = len(all_edges)
        if total_edges == 0:
            self.inter_connections = set()
            self.quantum_connection_count = sum(len(conns) for conns in self.intra_connections.values())
            self.classical_connection_count = 0
            print("No inter-edges generated.")
            return

        num_rewire = int(round(p * total_edges))
        # prepare global list of all possible directed inter-cluster edges for rewiring targets
        all_possible_inters = [
            (u, v)
            for a in range(self.num_clusters)
            for b in range(self.num_clusters)
            if a != b
            for u in self.clusters[a]
            for v in self.clusters[b]
        ]

        # perform rewiring: pick distinct indices to rewire
        rewire_indices = random.sample(range(total_edges), min(num_rewire, total_edges))
        for idx in rewire_indices:
            new_edge = random.choice(all_possible_inters)
            all_edges[idx] = new_edge

        # finalize: store as a set (unique). If duplicates exist by design and you want to preserve multiplicity,
        # you can store list separately (e.g., self.inter_connections_multiedges = all_edges)
        self.inter_connections = set(all_edges)

        # stats
        self.quantum_connection_count = sum(len(conns) for conns in self.intra_connections.values())
        self.classical_connection_count = len(self.inter_connections)
        print(f"Quantum Connections: {self.quantum_connection_count}")
        print(f"Classical Connections: {self.classical_connection_count} (target requested = {self.total_inter_connections}, generated = {total_edges}, unique stored = {len(self.inter_connections)})")
        print(f"Small-World (position-aligned offset rounds) with rewiring p = {p}")



    def _initialize_scale_free_network(self, gamma=2.5):
        """
        建立符合無尺度網路定義的 inter-connection，度分布服從 P(k) ∝ 1/k^gamma。
        inter 連線僅發生在跨 cluster 的節點對，並精確達到 total_inter_connections。
        """
        from collections import defaultdict

        self.inter_connections = set()
        self.intra_connections = defaultdict(set)

        # Step 1: cluster 內隨機連線（不動）
        for i in range(self.num_clusters):
            while len(self.intra_connections[i]) < self.intra_connections_per_cluster:
                src, tgt = random.sample(self.clusters[i], 2)
                self.intra_connections[i].add((src, tgt))

        # Step 2: 所有可能的 inter-cluster 邊
        allowed_pairs = []
        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                for u in self.clusters[i]:
                    for v in self.clusters[j]:
                        allowed_pairs.append((u, v))
                        allowed_pairs.append((v, u))

        if len(allowed_pairs) < self.total_inter_connections:
            raise ValueError("無法生成足夠跨 cluster 的 inter connections")

        # Step 3: 初始所有節點的度數設為 1（避免除以零）
        degree = {node: 1 for node in range(self.num_people)}

        # Step 4: 依據 P(k) ∝ k^gamma 做 weighted sampling
        def pair_weight(u, v, degree, gamma):
            return (degree[u] * degree[v]) ** (1 / gamma)

        selected_edges = set()
        while len(selected_edges) < self.total_inter_connections:
            weights = [pair_weight(u, v, degree, gamma) for u, v in allowed_pairs]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]

            idx = random.choices(range(len(allowed_pairs)), weights=probs, k=1)[0]
            u, v = allowed_pairs[idx]

            edge = (u, v)
            if edge not in selected_edges:
                selected_edges.add(edge)
                degree[u] += 1
                degree[v] += 1

        self.inter_connections = selected_edges

        # 統計
        self.quantum_connection_count = sum(len(conns) for conns in self.intra_connections.values())
        self.classical_connection_count = len(self.inter_connections)
        print(f"Quantum Connections: {self.quantum_connection_count}")
        print(f"Classical Connections: {self.classical_connection_count}")
        print(f"Scale-Free model generated with γ = {gamma}")

    def plot_degree_distribution(self, loglog=False, histtype='bar'):
        """
        繪製網路節點的連線度分布。
        
        Parameters:
            loglog (bool): 若為 True 則使用 log-log plot（適用於 scale-free 檢驗）
            histtype (str): 'bar' 或 'scatter'，顯示方式
        """
        import matplotlib.pyplot as plt
        from collections import Counter

        # 計算整張圖的 degree 分布（包含 inter + intra）
        G = nx.Graph()
        for conns in self.intra_connections.values():
            G.add_edges_from(conns)
        G.add_edges_from(self.inter_connections)

        degrees = [deg for _, deg in G.degree()]
        degree_count = Counter(degrees)
        x, y = zip(*sorted(degree_count.items()))

        plt.figure(figsize=(6, 4))
        if histtype == 'bar':
            plt.bar(x, y, color='skyblue', edgecolor='black')
        elif histtype == 'scatter':
            plt.plot(x, y, 'bo-')

        if loglog:
            plt.xscale('log')
            plt.yscale('log')
            plt.title("Degree Distribution (Log-Log)")
        else:
            plt.title("Degree Distribution")

        plt.xlabel("Degree (k)")
        plt.ylabel("Number of Nodes with Degree k")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    

    def _adjust_connection_count(self):
        """
        調整連線數量，確保與隨機網路相同，並顯示調整前後的差異
        """
        # 計算目前的連線數
        original_quantum_count = sum(len(conns) for conns in self.intra_connections.values())
        original_classical_count = len(self.inter_connections)

        # 目標連線數
        target_quantum = self.intra_connections_per_cluster * self.num_clusters
        target_classical = self.total_inter_connections

        print("\n=== 🔹 連線數調整前 🔹 ===")
        print(f"Quantum (區塊內) 原始連線數: {original_quantum_count} | 目標: {target_quantum}")
        print(f"Classical (區塊間) 原始連線數: {original_classical_count} | 目標: {target_classical}")

        # 如果量子連線數過多，隨機移除
        while original_quantum_count > target_quantum:
            cluster_id = random.choice(list(self.intra_connections.keys()))
            if self.intra_connections[cluster_id]:
                self.intra_connections[cluster_id].pop()
                original_quantum_count -= 1

        # 如果傳統連線數過多，隨機移除
        while original_classical_count > target_classical:
            if self.inter_connections:
                self.inter_connections.pop()
                original_classical_count -= 1

        # 如果連線數不足，隨機補充
        while original_quantum_count < target_quantum:
            cluster_id = random.choice(list(self.intra_connections.keys()))
            src, tgt = random.sample(self.clusters[cluster_id], 2)
            self.intra_connections[cluster_id].add((src, tgt))
            original_quantum_count += 1

        while original_classical_count < target_classical:
            src, tgt = random.sample(range(self.num_people), 2)
            self.inter_connections.add((src, tgt))
            original_classical_count += 1

        print("\n=== 🔹 連線數調整後 🔹 ===")
        print(f"Quantum (區塊內) 調整後連線數: {original_quantum_count} ✅")
        print(f"Classical (區塊間) 調整後連線數: {original_classical_count} ✅\n")        

    def run_simulation(self):
        """執行模擬"""
        for step in range(1, self.num_steps + 1):
            if self.method == "quantum":
                self._run_quantum_step()
            else:
                self._run_classical_step()

            if step % self.measure_interval == 0:
                self.infection_counts.append(sum(self.global_state))

        if self.save_csv:
            self.save_results()

    def _run_quantum_step(self):
        """量子方法的模擬步驟"""
        for cluster_id, nodes in self.intra_connections.items():
            cluster_nodes = self.clusters[cluster_id]
            num_qubits = len(cluster_nodes) + 1
            num_classical_bits = len(cluster_nodes)
            qc = QuantumCircuit(num_qubits, num_classical_bits)

            for idx, node in enumerate(cluster_nodes):
                if self.global_state[node] == 1:
                    qc.x(idx)

            aux_idx = len(cluster_nodes)
            for src, tgt in nodes:
                src_idx = cluster_nodes.index(src)
                tgt_idx = cluster_nodes.index(tgt)
                qc.rccx(src_idx, tgt_idx, aux_idx)
                qc.crx(self.infect, aux_idx, tgt_idx)
                qc.crx(-self.infect, src_idx, tgt_idx)
                qc.reset(aux_idx)
            good = 0
            for idx, node in enumerate(cluster_nodes):
                qc.cx(idx, aux_idx)
                qc.crx(-self.recover, aux_idx, idx)
                qc.reset(aux_idx)
                good = good +1
                print(idx)
                # if self.global_state[node] == 1 and random.random() < self.recover_rate:
                #     qc.reset(idx)

            qc.measure(range(len(cluster_nodes)), range(len(cluster_nodes)))
            compiled_circuit = transpile(qc, self.simulator)
            result = self.simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            measured_state = list(counts.keys())[0]

            for idx, node in enumerate(cluster_nodes):
                self.global_state[node] = int(measured_state[::-1][idx])

        # 區塊間傳染
        for src, tgt in self.inter_connections:
            if self.global_state[src] == 1 and random.random() < (np.sin(self.infect * np.pi / 2))**2:
                self.global_state[tgt] = 1
            elif self.global_state[tgt] == 1 and random.random() < (np.sin(self.infect * np.pi / 2))**2:
                self.global_state[src] = 1

    def _run_classical_step(self):
        """傳統方法的模擬步驟"""
        new_state = self.global_state.copy()

        # 區塊內傳染
        for cluster_id, connections in self.intra_connections.items():
            for src, tgt in connections:
                if self.global_state[src] == 1 and random.random() < self.infection_rate:
                    new_state[tgt] = 1
                elif self.global_state[tgt] == 1 and random.random() < self.infection_rate:
                    new_state[src] = 1

        # 區塊間傳染
        for src, tgt in self.inter_connections:
            if self.global_state[src] == 1 and random.random() < self.infection_rate:
                new_state[tgt] = 1
            elif self.global_state[tgt] == 1 and random.random() < self.infection_rate:
                new_state[src] = 1

        # 恢復過程
        good = 0
        for i in range(self.num_people):
            if self.global_state[i] == 1 and random.random() < self.recover_rate:
                new_state[i] = 0
            # good = good +1
            # print(good)

        self.global_state = new_state

    import matplotlib.pyplot as plt
    import networkx as nx

    def draw_network(self):

        G = nx.Graph()

        # 加入節點與顏色（依 cluster 分色）
        node_colors = {}
        ordered_nodes = []
        for i, cluster in enumerate(self.clusters):
            color = f"C{i}"
            for node in cluster:
                G.add_node(node)
                node_colors[node] = color
                ordered_nodes.append(node)  # 保留節點的整體順序（cluster 順序）

        # 加入量子連線（intra）
        for conns in self.intra_connections.values():
            for u, v in conns:
                G.add_edge(u, v, kind='quantum')

        # 加入古典連線（inter）
        for u, v in self.inter_connections:
            G.add_edge(u, v, kind='classical')

        # ▶️ 使用圓形排列（同 cluster 相鄰）
        num_nodes = len(ordered_nodes)
        angle_step = 2 * np.pi / num_nodes
        radius = 5
        pos = {}
        for i, node in enumerate(ordered_nodes):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[node] = (x, y)

        # 畫節點（依 cluster）
        for i, cluster in enumerate(self.clusters):
            nx.draw_networkx_nodes(G, pos,
                                nodelist=cluster,
                                node_color=node_colors[cluster[0]],
                                label=f"Cluster {i}",
                                node_size=100)

        # 畫邊：intra 為實線、inter 為虛線
        quantum_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "quantum"]
        classical_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "classical"]

        nx.draw_networkx_edges(G, pos, edgelist=quantum_edges, style='solid', edge_color='black', width=1.2)
        nx.draw_networkx_edges(G, pos, edgelist=classical_edges, style='dashed', edge_color='gray', width=1.0)

        # 可選：標記節點
        # nx.draw_networkx_labels(G, pos, font_size=6)

        plt.title("Cluster-Aligned Ring Network (Quantum: solid, Classical: dashed)")
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.show()


    
    def save_results(self):
        """儲存結果為 CSV，包含所有實驗參數"""
        # 產生唯一的檔案名稱，包含時間戳記和實驗參數
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"epidemic_{timestamp}_I{self.infection_rate}_R{self.recover_rate}_P{self.num_people}_C{self.num_clusters}_S{self.num_steps}.csv"
        
        # 設定儲存目錄
        full_path = os.path.join(self.results_dir, self.csv_filename)

        # 建立資料夾（如果尚未存在）
        os.makedirs(self.results_dir, exist_ok=True)

        # 建立 DataFrame，包含感染數據與所有實驗參數
        steps = list(range(self.measure_interval, self.num_steps + 1, self.measure_interval))
        data = {
            "Step": steps,
            "Infected Count": self.infection_counts,
            "Infection Rate": [self.infection_rate] * len(steps),
            "Recovery Rate": [self.recover_rate] * len(steps),
            "Total People": [self.num_people] * len(steps),
            "Initial Infected": [self.initial_infected] * len(steps),
            "Num Clusters": [self.num_clusters] * len(steps),
            "Intra-cluster Connections": [self.intra_connections_per_cluster] * len(steps),
            "Inter-cluster Connections": [self.total_inter_connections] * len(steps),
            "Total Steps": [self.num_steps] * len(steps),
            "Measure Interval": [self.measure_interval] * len(steps),
            "Method": [self.method] * len(steps),
        }

        df = pd.DataFrame(data)

        # 儲存至 CSV
        df.to_csv(full_path, index=False)
        print(f"Results saved to {full_path}")
        
    def plot_results(self):
        """繪製感染人數變化圖"""
        steps = list(range(self.measure_interval, self.num_steps + 1, self.measure_interval))
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.infection_counts, marker="o", linestyle="-")
        plt.xlabel("Step")
        plt.ylabel("Number of Infected People")
        plt.title(f"Epidemic Spread Simulation ({self.method.capitalize()})")
        plt.grid(True)
        plt.show()
