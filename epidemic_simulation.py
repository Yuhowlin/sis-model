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

    def _initialize_scale_free_network(self):
        """使用 Barabási–Albert 無尺度網絡生成"""
        # 計算 m 值，讓總連線數接近隨機網路
        m = max(1, self.total_inter_connections // self.num_people)
        m1 = max(1, 8*self.intra_connections_per_cluster*self.num_clusters // self.num_people)
        # 生成 BA 網路
        G = nx.barabasi_albert_graph(self.num_people, m)
        G1 = nx.barabasi_albert_graph(self.num_people//self.num_clusters, m1)
        # 將網絡轉為連線結構
        for i in range(self.num_clusters):
            while len(self.intra_connections[i]) < self.intra_connections_per_cluster:
                src, tgt = random.sample(self.clusters[i], 2)
                self.intra_connections[i].add((src, tgt))

        self.inter_connections = set(
            (u, v) for u, v in G.edges if any(u in cluster and v not in cluster for cluster in self.clusters)
        )

        # 調整總連線數，使其與隨機網路相等
        self._adjust_connection_count()

        # print(f"Quantum Connections: {self.quantum_connection_count}")
        # print(f"Classical Connections: {self.classical_connection_count}")
        # print('scale-free')

    def _initialize_small_world_network(self):
        """使用 Watts-Strogatz 小世界網絡生成"""
        # 計算 k 值，讓總連線數接近隨機網路
        k = max(2, 3*(self.total_inter_connections * 2) // self.num_people)
        k1 = max(2, (self.intra_connections_per_cluster * 2 * self.num_clusters) // self.num_people)
        # 生成 WS 網路 (p=0.3 控制隨機程度)
        G = nx.watts_strogatz_graph(self.num_people, k, 0.3)
        G1 = nx.watts_strogatz_graph(self.num_people//self.num_clusters, k1, 0.3)
        # 將網絡轉為連線結構
        for i in range(self.num_clusters):
            while len(self.intra_connections[i]) < self.intra_connections_per_cluster:
                src, tgt = random.sample(self.clusters[i], 2)
                self.intra_connections[i].add((src, tgt))

        self.inter_connections = set(
            (u, v) for u, v in G.edges if any(u in cluster and v not in cluster for cluster in self.clusters)
        )
            # 調整總連線數，使其與隨機網路相等
        self._adjust_connection_count()

            # print(f"Quantum Connections: {self.quantum_connection_count}")
            # print(f"Classical Connections: {self.classical_connection_count}")
            # print('small-world')

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

            for idx, node in enumerate(cluster_nodes):
                qc.cx(idx, aux_idx)
                qc.crx(-self.recover, aux_idx, idx)
                qc.reset(aux_idx)

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
        for i in range(self.num_people):
            if self.global_state[i] == 1 and random.random() < self.recover_rate:
                new_state[i] = 0

        self.global_state = new_state

    def compare_methods_with_stats(self, infection_rates, recover_rates, initial_infected_values, num_simulations=5):
        """比較量子 vs 傳統方法的影響 (多次模擬 + 計算平均與標準差)"""
        results = {"quantum": {}, "classical": {}}

        for infection_rate in infection_rates:
            for recover_rate in recover_rates:
                for method in ["quantum", "classical"]:
                    results[method][(infection_rate, recover_rate)] = {}
                    for initial_infected in initial_infected_values:
                        all_counts = []
                        for _ in range(num_simulations):
                            sim = EpidemicSimulation(
                                infection_rate, recover_rate, self.num_people,
                                initial_infected, self.num_clusters,
                                self.intra_connections_per_cluster, self.total_inter_connections,
                                self.num_steps, self.measure_interval, method
                            )
                            sim.run_simulation()
                            all_counts.append(sim.infection_counts)

                        avg_counts = np.mean(all_counts, axis=0)
                        std_counts = np.std(all_counts, axis=0)
                        results[method][(infection_rate, recover_rate)][initial_infected] = (avg_counts, std_counts)

        return results

    
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
