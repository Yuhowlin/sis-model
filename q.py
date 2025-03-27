from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 創建一個有2個量子位和2個經典位的量子電路
qc = QuantumCircuit(2, 2)

# 添加量子門
qc.h(0)          # 在第1個量子位上應用 Hadamard 門，創建疊加態
qc.cx(0, 1)      # 在第1個量子位（控制位）和第2個量子位（目標位）之間應用 CNOT 門，創建糾纏

# 測量量子位並將結果存入經典位
qc.measure([0, 1], [0, 1])

# 顯示量子電路
print(qc.draw())

# 使用 Qiskit 的 Aer 模擬器來模擬量子電路
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc, simulator, shots=1000)  # 設置模擬次數（shots）為1000
result = job.result()

# 獲取結果並可視化
counts = result.get_counts(qc)
print("\n測量結果:", counts)
plot_histogram(counts)
plt.show()
