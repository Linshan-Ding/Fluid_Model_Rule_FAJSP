import torch
import time
import os
import glob
from PPO import FAJSP_Environment, PolicyNetwork


def eval_single_model(pth):
    """评估单个模型文件"""
    try:
        nums = os.path.basename(pth).replace('.pth', '').split('_')[-6:]
        config = [int(n) for n in nums]

        env = FAJSP_Environment(M_p=config[0], M_a=config[1],
                                kind_count=config[3], product_count=config[2],
                                J_r=config[4], N_p=config[5])
        env.reset()

        data = torch.load(pth, map_location='cpu')
        params = data['policy'] if 'policy' in data else data.get('best_policy_state')

        for key in params:
            if 'atn.query.weight' in key:
                atn_dim = params[key].shape[1]
            if 'mlp.network.0.weight' in key:
                mlp_dim = params[key].shape[1]

        net = PolicyNetwork(atn_dim, mlp_dim)
        net.load_state_dict(params)
        net.eval()

        times = []
        while True:
            acts = env.get_candidate_actions()
            if not acts: break

            start = time.perf_counter()
            feats = {a: env.action_features(a)[a] for a in acts}
            action = acts[torch.argmax(net(env.current_state, acts, feats)).item()]
            times.append((time.perf_counter() - start) * 1000)
            env.step(action)

        if times:
            print(
                f"模型: {os.path.basename(pth):<30} | 平均时间: {sum(times) / len(times):6.2f}ms | 步骤: {len(times):3d} | CT: {env.completion_time}")

    except Exception as e:
        print(f"模型: {os.path.basename(pth):<30} | 错误: {str(e)[:40]}...")


def main():
    pth_files = glob.glob("*.pth")
    if not pth_files:
        print("未找到.pth文件")
        return

    print(f"找到 {len(pth_files)} 个模型文件:")
    for i, f in enumerate(pth_files, 1):
        print(f"  {i}. {f}")

    mode = input("\n运行模式 (1=单个, 2=全部, 其他=退出): ").strip()

    if mode == "1":
        try:
            idx = int(input(f"选择模型 (1-{len(pth_files)}): "))
            if 1 <= idx <= len(pth_files):
                eval_single_model(pth_files[idx - 1])
        except:
            print("无效选择")
    elif mode == "2":
        for f in pth_files:
            eval_single_model(f)


if __name__ == "__main__":
    # python - m visdom.server
    main()