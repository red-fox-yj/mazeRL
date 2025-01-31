import os
import sys
import numpy as np
import random
import pickle
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw

def load_map_from_file(map_file):
    """
    假设地图里每行使用空格分隔，
    每个元素可能是:
      '.' 空地
      '#' 墙
      'S' 起点
      'G' 终点
      'T' 陷阱
      'K1','K2',... 钥匙
      'D1','D2',... 门
    """
    with open(map_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    maze_map = []
    for line in lines:
        row = line.split()
        maze_map.append(row)
    return maze_map


class MazeEnvMultiKeyDoor:
    """
    多钥匙、多门环境，且状态包含：
      (row, col, frozenset(keys_in_inventory), frozenset(doors_opened))
    """
    def __init__(self, map_file):
        self.map_file = map_file
        self.original_map = load_map_from_file(map_file)
        self.n_rows = len(self.original_map)
        self.n_cols = len(self.original_map[0]) if self.n_rows > 0 else 0

        # 动作空间
        self.action_space = [0,1,2,3]  # 0=上,1=下,2=左,3=右

        self.start_pos = None
        self.goal_pos  = None
        self.trap_positions = []
        
        # 记录门和钥匙的位置信息: {(r,c): "1"} => 表示 D1 或 K1
        self.door_positions = {}  
        self.key_positions  = {}

        # 扫描地图，提取特殊元素位置
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cell = self.original_map[r][c]
                if cell == 'S':
                    self.start_pos = (r, c)
                elif cell == 'G':
                    self.goal_pos = (r, c)
                elif cell == 'T':
                    self.trap_positions.append((r, c))
                elif cell.startswith('K'):
                    key_id = cell[1:]  # K1 => "1"
                    self.key_positions[(r,c)] = key_id
                elif cell.startswith('D'):
                    door_id = cell[1:] # D1 => "1"
                    self.door_positions[(r,c)] = door_id

        if not self.start_pos:
            raise ValueError("地图中没有起点 S!")
        if not self.goal_pos:
            raise ValueError("地图中没有终点 G!")

        self.reset()

    def reset(self):
        """
        回合开始:
        - 智能体回到起点
        - 清空已经持有的钥匙
        - 清空已打开的门
        """
        self.agent_pos = self.start_pos
        self.inventory = set()       # 已拿到的钥匙ID
        self.doors_opened = set()    # 已打开的门ID (如 {'1','2'})
        
        return self._get_state()

    def _get_state(self):
        """
        状态: (row, col, frozenset(self.inventory), frozenset(self.doors_opened))
        """
        r, c = self.agent_pos
        return (r, c, frozenset(self.inventory), frozenset(self.doors_opened))

    def step(self, action):
        """
        action: 0=上,1=下,2=左,3=右
        return: (next_state, reward, done, info)
        """
        r, c = self.agent_pos
        nr, nc = r, c
        if action == 0:  # up
            nr = r - 1
        elif action == 1:  # down
            nr = r + 1
        elif action == 2:  # left
            nc = c - 1
        elif action == 3:  # right
            nc = c + 1

        reward = 0.0
        done   = False
        info = {
            "key_picked": None,
            "door_opened": None,
            "trap_triggered": False
        }

        # -- 撞墙 or 越界 --
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            reward -= 1.0
            nr, nc = r, c
        elif self.original_map[nr][nc] == '#':
            reward -= 1.0
            nr, nc = r, c
        else:
            # ------ 如果那里是门 ------
            if (nr, nc) in self.door_positions:
                door_id = self.door_positions[(nr, nc)]
                # 检查是否已打开
                if door_id in self.doors_opened:
                    # 已打开 => 可以直接通过
                    self.agent_pos = (nr, nc)
                else:
                    # 未打开 => 检查是否有对应钥匙
                    if door_id in self.inventory:
                        # 有钥匙 => 开门
                        self.doors_opened.add(door_id)
                        reward += 3.0
                        info["door_opened"] = door_id
                        self.agent_pos = (nr, nc)
                    else:
                        # 没有钥匙 => 无法通过
                        reward -= 2.0
                        nr, nc = r, c
            # ------ 如果是钥匙 ------
            elif (nr, nc) in self.key_positions:
                key_id = self.key_positions[(nr, nc)]
                self.agent_pos = (nr, nc)
                # 如果还没拿过该钥匙
                if key_id not in self.inventory:
                    self.inventory.add(key_id)
                    reward += 5.0
                    info["key_picked"] = key_id
            else:
                # ------ 普通空地 ------
                self.agent_pos = (nr, nc)

                # 踩到陷阱?
                if (nr, nc) in self.trap_positions:
                    reward -= 5.0
                    info["trap_triggered"] = True
                    # 回到起点 & 清空背包 & 重新关门
                    self.agent_pos = self.start_pos
                    self.inventory.clear()
                    self.doors_opened.clear()
                    nr, nc = self.start_pos

            # 到达终点?
            if (nr, nc) == self.goal_pos:
                reward += 10.0
                done = True

        # 每步微小负奖励
        reward -= 0.01

        next_state = self._get_state()
        return next_state, reward, done, info


# ------------------------------------------------------------------
# Q-learning 训练
# ------------------------------------------------------------------

def q_learning_train(env, episodes=500, alpha=0.1, gamma=0.99,
                     epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    """
    Q: dict[state] = [Q(s,a0), Q(s,a1), Q(s,a2), Q(s,a3)]
    state = (r, c, frozenset(keys), frozenset(doors_opened))
    """
    Q = {}
    def get_qvals(state):
        if state not in Q:
            Q[state] = [0.0, 0.0, 0.0, 0.0]
        return Q[state]

    rewards_history = []

    for ep in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                qvals = get_qvals(state)
                action = np.argmax(qvals)

            next_state, reward, done, _ = env.step(action)

            qvals = get_qvals(state)
            next_qvals = get_qvals(next_state)
            td_target = reward + (0 if done else gamma * max(next_qvals))
            qvals[action] += alpha * (td_target - qvals[action])

            state = next_state
            ep_reward += reward

        rewards_history.append(ep_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return Q, rewards_history

# ------------------------------------------------------------------
# 可视化 & 测试
# ------------------------------------------------------------------

def render_frame(env, img_size=300):
    """
    将当前环境渲染为图像：
    - 对于门，若已在 doors_opened 中，就换种颜色/字符显示
    - 对于钥匙，若在 inventory，就不再显示地图上
    """
    color_map = {
        'S': (0, 255, 0),  # 起点
        'G': (255, 0, 0),  # 终点
        'T': (255, 128, 0),# 陷阱
        '#': (0, 0, 0),    # 墙
        '.': (255,255,255)
    }
    h, w = env.n_rows, env.n_cols
    cell_size = img_size // max(h, w)
    img = Image.new("RGB", (w*cell_size, h*cell_size), (255,255,255))
    draw = ImageDraw.Draw(img)

    # 逐格绘制
    for r in range(h):
        for c in range(w):
            cell_str = env.original_map[r][c]
            fill_color = color_map.get(cell_str, (200,200,200))

            if cell_str.startswith('D'):
                # 是门，检查门ID是否在 opened 集合
                door_id = cell_str[1:]
                if door_id in env.doors_opened:
                    # 已打开 => 浅棕色
                    fill_color = (180,140,100)
                else:
                    # 未打开 => 深棕
                    fill_color = (128,76,19)
            elif cell_str.startswith('K'):
                # 是钥匙
                key_id = cell_str[1:]
                if key_id in env.inventory:
                    # 已拾取 => 不再显示地图上的钥匙
                    cell_str = '.'
                    fill_color = (255,255,255)
                else:
                    fill_color = (255,255,0)  # 黄色
            elif cell_str == 'S':
                fill_color = (0,255,0)
            elif cell_str == 'G':
                fill_color = (255,0,0)
            elif cell_str == 'T':
                fill_color = (255,128,0)
            elif cell_str == '#':
                fill_color = (0,0,0)
            elif cell_str == '.':
                fill_color = (255,255,255)

            draw.rectangle(
                [c*cell_size, r*cell_size, (c+1)*cell_size, (r+1)*cell_size],
                fill=fill_color
            )
            # 在格子上显示字符(可选)
            text_x = c*cell_size + cell_size//4
            text_y = r*cell_size + cell_size//4
            if cell_str not in ('.','#'):
                draw.text((text_x, text_y), cell_str, fill=(0,0,0))

    # 画智能体
    ar, ac = env.agent_pos
    agent_color = "purple" if len(env.inventory) > 0 else "blue"
    draw.text(
        (ac*cell_size + cell_size//3, ar*cell_size + cell_size//4),
        "A", fill=agent_color
    )
    return np.array(img)


def test_agent(env, Q, max_steps=100, gif_name=None, fps=2):
    """
    使用给定 Q 表测试，并选项地保存动图
    """
    frames = []
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    frames.append(render_frame(env))
    while not done and step_count < max_steps:
        qvals = Q.get(state, [0,0,0,0])
        action = np.argmax(qvals)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        step_count += 1

        frames.append(render_frame(env))

    # 保存gif
    if gif_name:
        imageio.mimsave(gif_name, frames, fps=fps)
        print(f"Animation saved to {gif_name}")

    return {
        "reward": total_reward,
        "steps": step_count,
        "done": done
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <map_file>")
        sys.exit(1)

    map_file = sys.argv[1]
    if not os.path.exists(map_file):
        print(f"地图文件不存在: {map_file}")
        sys.exit(1)

    # 输出目录
    os.makedirs("outputs", exist_ok=True)
    base = os.path.basename(map_file)
    map_name, _ = os.path.splitext(base)
    q_table_path = f"outputs/{map_name}_Qtable.pkl"
    gif_path     = f"outputs/{map_name}_test.gif"

    env = MazeEnvMultiKeyDoor(map_file)
    print(f"加载地图: {map_file}, 大小: {env.n_rows}x{env.n_cols}")

    # 如果已有模型，则跳过训练
    if os.path.exists(q_table_path):
        print(f"检测到已有 Q 表: {q_table_path}，跳过训练...")
        with open(q_table_path, 'rb') as f:
            Q = pickle.load(f)
    else:
        print("开始训练...")
        Q, rewards_history = q_learning_train(env, episodes=800, alpha=0.1, gamma=0.99,
                                              epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
        with open(q_table_path, 'wb') as f:
            pickle.dump(Q, f)
        print(f"Q 表已保存到 {q_table_path}")

    # 测试
    result = test_agent(env, Q, max_steps=200, gif_name=gif_path, fps=2)
    print("测试结果:", result)