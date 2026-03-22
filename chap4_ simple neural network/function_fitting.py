import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ==========================================
# 1. 定义目标函数
# ==========================================
def target_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

# ==========================================
# 2. 数据采集 (加入 Shuffle)
# ==========================================
def generate_data(num_samples=300, noise_std=0.15):
    # 生成所有数据
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_samples).reshape(-1, 1)
    y_true = target_function(x)
    y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)
    
    # 创建索引并打乱
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # 按照打乱后的索引重新排列 x 和 y
    x_shuffled = x[indices]
    y_shuffled = y_noisy[indices]
    y_true_shuffled = y_true[indices] # 对应的真实值也要跟着动（虽然测试时我们只用无噪的真实值逻辑，但这里为了对应方便）
    
    # 划分训练集 (80%) 和 测试集 (20%)
    split_idx = int(0.8 * num_samples)
    
    train_x = x_shuffled[:split_idx]
    train_y = y_shuffled[:split_idx]
    
    test_x = x_shuffled[split_idx:]
    # 测试集使用无噪声的真实值 (注意：这里要用打乱后对应的真实值)
    test_y = y_true_shuffled[split_idx:] 
    
    return train_x, train_y, test_x, test_y

# ==========================================
# 3. 神经网络组件 (Tanh)
# ==========================================

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.05):
        self.lr = learning_rate
        # Xavier 初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, X, y_true):
        batch_size = X.shape[0]
        delta_output = (self.a2 - y_true) * (2.0 / batch_size)
        dW2 = np.dot(self.a1.T, delta_output)
        db2 = np.sum(delta_output, axis=0, keepdims=True)
        
        delta_hidden = np.dot(delta_output, self.W2.T) * tanh_derivative(self.z1)
        dW1 = np.dot(X.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X_train, y_train, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = np.mean((y_pred - y_train) ** 2)
            
            if np.isnan(loss):
                print(f"NaN at epoch {epoch}")
                break
                
            loss_history.append(loss)
            self.backward(X_train, y_train)
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.6f}")
        return loss_history

    def predict(self, X):
        return self.forward(X)

# ==========================================
# 4. 主程序
# ==========================================

if __name__ == "__main__":
    HIDDEN_UNITS = 50
    LEARNING_RATE = 0.05 
    EPOCHS = 20000  
    
    print("正在生成数据 (已打乱)...")
    train_x, train_y, test_x, test_y = generate_data(num_samples=300, noise_std=0.15)
    
    # --- 数据归一化 ---
    x_mean = np.mean(train_x)
    x_std = np.std(train_x)
    if x_std == 0: x_std = 1.0
    
    train_x_norm = (train_x - x_mean) / x_std
    test_x_norm = (test_x - x_mean) / x_std
    
    # 绘图用密集点 (覆盖整个区间)
    x_dense = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    x_dense_norm = (x_dense - x_mean) / x_std
    
    print(f"归一化完成 (Mean: {x_mean:.2f}, Std: {x_std:.2f})")
    print(f"训练集 X 范围: [{train_x.min():.2f}, {train_x.max():.2f}]")
    print(f"测试集 X 范围: [{test_x.min():.2f}, {test_x.max():.2f}]")
    print("开始训练...")
    
    model = SimpleNN(input_dim=1, hidden_dim=HIDDEN_UNITS, output_dim=1, learning_rate=LEARNING_RATE)
    losses = model.train(train_x_norm, train_y, epochs=EPOCHS)
    
    print("训练完成，评估中...")
    pred_test = model.predict(test_x_norm)
    y_dense_pred = model.predict(x_dense_norm)
    y_dense_true = target_function(x_dense)
    
    test_mse = np.mean((pred_test - test_y) ** 2)
    print(f"测试集 MSE: {test_mse:.6f}")
    
    if test_mse < 0.05:
        print("✅ 成功！模型完美拟合。")
    else:
        print("⚠️ 误差仍较大，请检查。")

    # --- 绘图 ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_dense, y_dense_true, 'k-', linewidth=2, label='True Function')
    plt.plot(x_dense, y_dense_pred, 'r--', linewidth=2, label='NN Prediction')
    
    plt.scatter(train_x, train_y, c='blue', s=15, alpha=0.4, label='Train Data (Shuffled)')
    plt.scatter(test_x, test_y, c='green', s=15, alpha=0.8, label='Test Data')
    
    plt.title(f'Function Fitting (Shuffled Data)\nTest MSE: {test_mse:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, 'b-')
    plt.title(f'Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fitting_result_shuffled.png')
    print("图片已保存：fitting_result_shuffled.png")
    plt.show()