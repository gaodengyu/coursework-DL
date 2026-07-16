import networkx as nx
import matplotlib.pyplot as plt

def draw_safe_fsm_simple():
    # 创建有向图
    G = nx.DiGraph()

    # 定义 5 个核心状态 
    states = {
        "Locked": "Locked\n(Initial)",
        "Revealed": "Panel\nRevealed",
        "Alarm": "Alarm",
        "Open": "Open",
        "Closed": "Panel\nClosed"
    }
    
    # 添加状态转换 (Edges) [cite: 240, 250, 251]
    # 格式: (起点, 终点, 标签)
    edges = [
        ("Locked", "Revealed", "openPanel()\n[key inserted]"),
        ("Revealed", "Open", "validate(pw)\n[correct]"),
        ("Revealed", "Alarm", "validate(pw)\n[incorrect]"),
        ("Alarm", "Open", "validate(pw)\n[correct]"),
        ("Open", "Closed", "closePanel()\n[door closed & buttoned]"),
        ("Closed", "Locked", "lockSafe()\n[key removed]")
    ]

    for start, end, label in edges:
        G.add_edge(start, end, label=label)

    # 手动设置节点位置，使其布局美观
    pos = {
        "Locked": (0, 2),
        "Revealed": (2, 2),
        "Alarm": (2, 0),
        "Open": (4, 2),
        "Closed": (2, 4)
    }

    plt.figure(figsize=(12, 8))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=states, font_size=10, font_weight='bold')

    # 绘制边
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', connectionstyle="arc3,rad=0.1")
    
    # 绘制边上的文字标签
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    plt.title("Question 7: Electronic Safe Lock FSM", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_safe_fsm_simple()