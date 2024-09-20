import numpy as np
import matplotlib.pyplot as plt
import logging
import random

# 固定 NumPy 和 Python 的随机种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Binary Mechanism B
class BinaryMechanism:
    def __init__(self, T, epsilon, sigma):
        self.T = T
        self.epsilon = epsilon
        # 确保 sigma 的长度与 T 一致
        if len(sigma) != T:
            print(sigma,T)
            raise ValueError(f"Length of sigma ({len(sigma)}) must be equal to T ({T})")
        self.sigma = sigma
        self.alpha = np.zeros(int(np.ceil(np.log2(max(T, 2)))) + 1)
        self.alpha_hat = np.zeros(int(np.ceil(np.log2(max(T, 2)))) + 1)
        self.epsilon_prime = epsilon / np.log(max(T, 2))
        self.max_noise_count = 0  # 用于记录最大噪声添加次数


    
    def laplace_noise(self, scale):
        return np.random.laplace(0, scale)

    def run(self):
        B_t = []
        noise_count = 0  # 局部变量记录 lap(1/epsilon_prime) 噪声次数
        for t in range(1, self.T + 1):
            bin_t = list(map(int, bin(t)[2:]))[::-1]  # Get binary representation in reverse order
            i = min([j for j in range(len(bin_t)) if bin_t[j] != 0])  # Get min i where Bin_j(t) != 0

            # Update alpha_i
            self.alpha[i] = sum(self.alpha[j] for j in range(i)) + self.sigma[t-1]
            
            # Reset alpha_j and alpha_hat_j for j < i
            for j in range(0, i):
                self.alpha[j] = 0
                self.alpha_hat[j] = 0
            
            # Compute noisy alpha_hat_i
            self.alpha_hat[i] = self.alpha[i] + self.laplace_noise(1 / self.epsilon_prime)
            noise_count += 1  # 每次调用 laplace_noise 就累加一次
            self.max_noise_count = max(self.max_noise_count, noise_count)  # 更新最大噪声添加次数

            # Output the estimate at time t
            B_t.append(sum(self.alpha_hat[j] for j in range(len(bin_t)) if bin_t[j] == 1))

        return B_t, self.max_noise_count  # 返回 B_t 和最大噪声添加次数


# 修改后的 Thompson Sampling with Private Mechanism and extra step for empirical mean
class PrivateThompsonSamplingBanditWithCorrection_M:
    def __init__(self, n_arms, epsilon, env):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.arm_counts = np.zeros(n_arms)
        self.reward_sums = np.zeros(n_arms)
        self.epsilon = epsilon
        self.phi = [[] for _ in range(n_arms)]  # 每个臂的phi集合
        self.C = np.zeros(n_arms)  # 每个臂的 C_j 初始化
        self.B = np.zeros(n_arms)  # 存储每个臂的 Binary Mechanism 值
        self.r = np.zeros(n_arms, dtype=int)  # 记录每个臂的 r 值
        self.current_time = 0
        self.C_updates = np.zeros(n_arms, dtype=int)  # 记录C_j更新次数
        self.B_updates = np.zeros(n_arms, dtype=int)  # 记录B_j更新次数
        self.noise_epsilon_count = 0  # 记录 lap(1/epsilon) 的添加次数
        self.noise_epsilon_prime_count = 0  # 记录添加 lap(1/epsilon_prime) 的次数
        self.max_noise_updates = 0  # 记录最大噪声更新次数

        # 初始化时拉动每个臂一次
        '''
        for arm in range(n_arms):
            reward = env.pull_arm(arm)
            self.update(arm, reward)
        '''

    def laplace_noise(self, scale):
        return np.random.laplace(0, scale)

    def select_arm(self):
        sampled_theta = np.random.beta(self.alpha+1, self.beta+1)
        return np.argmax(sampled_theta)

    def run_binary_mechanism(self, chosen_arm):
        # T = armj当前被拉总次数 - 2^(r+1)+1
        T = int(self.arm_counts[chosen_arm] - 2**(self.r[chosen_arm] + 1) +1)
        sigma = np.array(self.phi[chosen_arm])

        bm = BinaryMechanism(T, self.epsilon, sigma)
        B_t, max_noise_count = bm.run()  # 获取 B_t 和最大噪声添加次数


        # 记录最大噪声更新次数
        self.max_noise_updates = max(self.max_noise_updates, max_noise_count)
        self.B_updates[chosen_arm] += 1
 
        # 返回最后的 B_t 值
        return B_t[-1] if B_t else 0

    def update(self, chosen_arm, reward):
        self.current_time += 1  # 更新当前时间
        self.arm_counts[chosen_arm] += 1
            
        O_j = self.arm_counts[chosen_arm]
        t = self.current_time
        #print(t)
        #print(O_j)
        if self.arm_counts[chosen_arm] == 1:
            # 初始化阶段，C_j 初始化为 reward + Laplace 噪声，B_j 为 0
            self.C[chosen_arm] = reward + self.laplace_noise(1 / self.epsilon)
            self.B[chosen_arm] = 0
            self.noise_epsilon_count += 1  # 记录 lap(1/epsilon) 添加次数
            self.C_updates[chosen_arm] += 1  # 更新C_j计数
        else:
            self.phi[chosen_arm].append(reward)

            # 检查是否达到 2^（r+2） - 1 次拉动次数
            if self.arm_counts[chosen_arm] == 2**(self.r[chosen_arm] + 2) - 1:
                self.C[chosen_arm] = sum(self.phi[chosen_arm]) + self.laplace_noise(1 / self.epsilon)
                self.phi[chosen_arm] = []  # 重置 phi
                self.r[chosen_arm] += 1
                self.C_updates[chosen_arm] += 1  # 更新C_j计数
                self.noise_epsilon_count += 1  # 记录 lap(1/epsilon) 添加次数

                # 将 B_j 重置为 0
                self.B[chosen_arm] = 0
            else:
                self.B[chosen_arm] = self.run_binary_mechanism(chosen_arm)
                #self.B_updates[chosen_arm] += 1  # 更新B_j计数

        # 经验均值的修正
        total_feedback = self.C[chosen_arm] + self.B[chosen_arm]
        empirical_mean = total_feedback / O_j
        correction = 6 * np.sqrt(8 * np.log(O_j + 1) * np.log(t)) / (self.epsilon * O_j)
        empirical_mean_with_correction = max(0, min(empirical_mean + correction, 1))

        # 更新 alpha 和 beta
        self.alpha[chosen_arm] = max(empirical_mean_with_correction * O_j, 1e-6)
        self.beta[chosen_arm] = max((1 - empirical_mean_with_correction) * O_j, 1e-6)

class PrivateThompsonSamplingBanditWithCorrection_O:
    def __init__(self, n_arms, epsilon, env):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.arm_counts = np.zeros(n_arms)
        self.reward_sums = np.zeros(n_arms)
        self.epsilon = epsilon
        self.phi = [[] for _ in range(n_arms)]  # 每个臂的phi集合
        self.C = np.zeros(n_arms)  # 每个臂的 C_j 初始化
        self.B = np.zeros(n_arms)  # 存储每个臂的 Binary Mechanism 值
        self.r = np.zeros(n_arms, dtype=int)  # 记录每个臂的 r 值
        self.current_time = 0
        self.C_updates = np.zeros(n_arms, dtype=int)  # 记录C_j更新次数
        self.B_updates = np.zeros(n_arms, dtype=int)  # 记录B_j更新次数
        self.noise_epsilon_count = 0  # 记录 lap(1/epsilon) 的添加次数
        self.noise_epsilon_prime_count = 0  # 记录添加 lap(1/epsilon_prime) 的次数
        self.max_noise_updates = 0  # 记录最大噪声更新次数
        # 初始化时拉动每个臂一次
        '''for arm in range(n_arms):
            reward = env.pull_arm(arm)
            self.update(arm, reward)
        '''

    def laplace_noise(self, scale):
        return np.random.laplace(0, scale)

    def select_arm(self):
        sampled_theta = np.random.beta(self.alpha+1, self.beta+1)
       # print(sampled_theta)
        return np.argmax(sampled_theta)

    def run_binary_mechanism(self, chosen_arm):
        # T = armj当前被拉总次数 - 2^(r)
        T = int(self.arm_counts[chosen_arm] - 2**(self.r[chosen_arm]))
        sigma = np.array(self.phi[chosen_arm])

        bm = BinaryMechanism(T, self.epsilon, sigma)
        B_t, max_noise_count = bm.run()  # 获取 B_t 和最大噪声添加次数


        # 记录最大噪声更新次数
        self.max_noise_updates = max(self.max_noise_updates, max_noise_count)

        self.B_updates[chosen_arm] += 1
        
        # 返回最后的 B_t 值
        return B_t[-1] if B_t else 0

    def update(self, chosen_arm, reward):
        self.current_time += 1  # 更新当前时间
        self.arm_counts[chosen_arm] += 1
        O_j = self.arm_counts[chosen_arm]
        t = self.current_time

        if self.arm_counts[chosen_arm] == 1:
            # 初始化阶段，C_j 初始化为 reward + Laplace 噪声，B_j 为 0
            self.C[chosen_arm] = reward + self.laplace_noise(1 / self.epsilon)
            self.B[chosen_arm] = 0
            self.noise_epsilon_count += 1  # 记录 lap(1/epsilon) 添加次数

            self.C_updates[chosen_arm] += 1  # 更新C_j计数
        else:
            self.phi[chosen_arm].append(reward)

            # 检查是否达到 2^（r）  次拉动次数
            if self.arm_counts[chosen_arm] == 2**(self.r[chosen_arm]+1):
                self.C[chosen_arm] = sum(self.phi[chosen_arm]) + self.laplace_noise(1 / self.epsilon)
                self.phi[chosen_arm] = []  # 重置 phi
                self.r[chosen_arm] += 1
                self.C_updates[chosen_arm] += 1  # 更新C_j计数
                self.noise_epsilon_count += 1  # 记录 lap(1/epsilon) 添加次数


                # 将 B_j 重置为 0
                self.B[chosen_arm] = 0
            else:
                self.B[chosen_arm] = self.run_binary_mechanism(chosen_arm)
               # self.B_updates[chosen_arm] += 1  # 更新B_j计数

        # 经验均值的修正
        total_feedback = self.C[chosen_arm] + self.B[chosen_arm]
        empirical_mean = total_feedback / O_j
        correction = 6 * np.sqrt(8 * np.log(O_j + 1) * np.log(t)) / (self.epsilon * O_j)
        empirical_mean_with_correction = max(0, min(empirical_mean + correction, 1))

        # 更新 alpha 和 beta
        self.alpha[chosen_arm] = max(empirical_mean_with_correction * O_j, 1e-6)
        self.beta[chosen_arm] = max((1 - empirical_mean_with_correction) * O_j, 1e-6)



# 模拟多臂老虎机环境
class MultiArmedBanditEnv:
    def __init__(self, true_probabilities):
        self.true_probabilities = true_probabilities

    def pull_arm(self, arm):
        return np.random.binomial(1, self.true_probabilities[arm])

# 测试函数
def test_private_thompson_sampling_with_correction():
    # 设置随机种子
  #  seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
   # epsilons = [0.1, 1, 10, 50, 100, 500, 1000]
    seeds = [42]
    epsilons  = [500]
    n_arms = 5
    true_probabilities = [0.75, 0.625, 0.5, 0.375, 0.25]
    n_rounds = [2000]
    
    for seed in seeds:
        for epsilon in epsilons:
            # 设置随机种子
            set_seed(seed)   
            env = MultiArmedBanditEnv(true_probabilities)
            ts_bandit_M = PrivateThompsonSamplingBanditWithCorrection_M(n_arms, epsilon, env)
            total_rewards_M = np.zeros(n_rounds)
            chosen_arms_M = np.zeros(n_rounds)
            cumulative_regret_M = np.zeros(n_rounds)  # 用于记录每轮的累积后悔值
            cumulative_reward_M = 0
            ts_bandit_O = PrivateThompsonSamplingBanditWithCorrection_O(n_arms, epsilon,env)
            total_rewards_O = np.zeros(n_rounds)
            chosen_arms_O = np.zeros(n_rounds)
            cumulative_regret_O = np.zeros(n_rounds)  # 用于记录每轮的累积后悔值
            cumulative_reward_O = 0
            optimal_arm = np.argmax(true_probabilities)
            optimal_arm_reward = true_probabilities[optimal_arm]

            for n in n_rounds:
                t=0
                while(t<n):
                    if(t==0):
                       for arm in range(n_arms):
                            chosen_arms_M[t] = arm
                            chosen_arms_O[t]=arm
                            regret_M = optimal_arm_reward - true_probabilities[arm] 
                            cumulative_regret_M[t] = regret_M 
                            regret_O = optimal_arm_reward - true_probabilities[arm]
                            cumulative_regret_O[t] = regret_O
                            reward = env.pull_arm(arm)
                            ts_bandit_M.update(arm, reward)
                            print(ts_bandit_M.current_time)
                            reward_O = env.pull_arm(arm)
                            ts_bandit_O.update(arm, reward_O)    
                            t=ts_bandit_M.current_time   
                    else:                    
                        chosen_arm_M = ts_bandit_M.select_arm()
                        #print(chosen_arm_M)
                        reward_M = env.pull_arm(chosen_arm_M)
                        #print(reward_M)
                        ts_bandit_M.update(chosen_arm_M, reward_M)
                        #print(ts_bandit_M.noise_epsilon_count)
                        total_rewards_M[t] = reward_M
                        chosen_arms_M[t] = chosen_arm_M
                        
                        # 计算后悔值：最优臂回报 - 实际选择臂的回报
                        regret_M = optimal_arm_reward - true_probabilities[chosen_arm_M]
                        cumulative_regret_M[t] = regret_M
                        
                        
                        chosen_arm_O = ts_bandit_O.select_arm()
                        reward_O = env.pull_arm(chosen_arm_O)
                        ts_bandit_O.update(chosen_arm_O, reward_O)
                        total_rewards_O[t] = reward_O
                        chosen_arms_O[t] = chosen_arm_O
                        
                        # 计算后悔值：最优臂回报 - 实际选择臂的回报
                        regret_O = optimal_arm_reward - true_probabilities[chosen_arm_O]
                        cumulative_regret_O[t] = regret_O
                        t=t+1
                        if t % 100 == 0:
                            print(f"Seed: {seed}, Epsilon: {epsilon}, 当前的次数: {t}", flush=True)

            # 计算累积遗憾
            regrets_M = np.cumsum(cumulative_regret_M)
            regrets_O = np.cumsum(cumulative_regret_O)

            cumulative_reward_M = np.sum(total_rewards_M)
            cumulative_reward_O = np.sum(total_rewards_O) 
            
            arm_selections_M = np.bincount(chosen_arms_M.astype(int))
            arm_selections_O = np.bincount(chosen_arms_O.astype(int))
        
            optimal_arm = np.argmax(true_probabilities)
            print(f"Optimal arm: {optimal_arm}")
            print(f"number of rounds:{n}")
            print(f"Total cumulative reward of Modified version: {cumulative_reward_M}")
            print(f"Total cumulative regret of Modified version: {regrets_M}")
            print(f"Arm selections of Modified version: {arm_selections_M}")
            print(f"Optimal arm selection count of Modified version: {arm_selections_M[optimal_arm]} out of {n_rounds} rounds")
            print(f"The number of times log noise added of Modified version: ", ts_bandit_M.noise_epsilon_count)
            print(f"The number of times binary noise added of Modified version: ",ts_bandit_M.max_noise_updates)
            print(f"Total cumulative reward of Original version: {cumulative_reward_O}")
            print(f"Total cumulative regret of Original version: {regrets_O}")
            print(f"Arm selections of Original version: {arm_selections_O}")
            print(f"Optimal arm selection count of Original version: {arm_selections_O[optimal_arm]} out of {n_rounds} rounds")
            print(f"The number of times log noise added of Original version: ",ts_bandit_O.noise_epsilon_count)
            print(f"The number of times binary noise added of Original version: ",ts_bandit_O.max_noise_updates)

        
            # 绘制累积遗憾的曲线
            plt.figure(figsize=(12, 8))
            plt.plot(regrets_M, label='Modified Version')
            plt.plot(regrets_O, label='Original Version')

            plt.xlabel('Rounds')
            plt.ylabel('Cumulative Regret')
            plt.title(f'Cumulative Regret over Time (Seed={seed}, Epsilon={epsilon})')
            plt.legend()
            plt.show()
            
            # 保存图像文件，文件名包含 seed 和 epsilon 值
            #plt.savefig(f"compare_seed_{seed}_epsilon_{epsilon}.png")
            plt.close()
           

# 调用测试函数
test_private_thompson_sampling_with_correction()
