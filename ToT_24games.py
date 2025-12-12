import asyncio
import os
import dotenv
from openai import AsyncOpenAI
from typing import List

dotenv.load_dotenv()


class ToTNode:
    def __init__(self, current_numbers, history, parent=None):
        self.current_numbers = current_numbers
        self.history = history
        self.parent = parent
        self.value = 0.0
        self.is_terminal = len(current_numbers) == 1

    def __repr__(self):
        return f"State: {self.current_numbers} | Val: {self.value}"


class GeneralAgent:
    def __init__(self, model: str = None, apikey: str = None, base_url: str = None):
        self.model = model or os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
        self.api_key = apikey or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def think(self, messages: List[dict], temperature: float = 0.7) -> str:
        """
        这里采用异步实现，可以同时执行思维树节点的评估判断
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return ""

class ToTAgent(GeneralAgent):
    def __init__(self, model: str = None, beam_width: int = 100):
        # 基于广度优先算法，beam_width是每轮选取的最大广度节点。（原论文采用的5，这里根据后面逻辑，选择默认100）
        super().__init__(model=model)
        self.beam_width = beam_width

    # --- 辅助函数：数字格式化 ---
    def _fmt(self, num):
        """将 1.0 转为 1，但保留 1.5"""
        if abs(num - round(num)) < 1e-5:
            return str(int(round(num)))
        return str(round(num, 4)) # 保留几位小数避免无限长

    def _get_next_steps(self, node: ToTNode) -> List[ToTNode]:
        nums = node.current_numbers
        if len(nums) < 2: return []
        next_nodes = []

        for i in range(len(nums)):
            for j in range(len(nums)):
                if i == j :continue

                a, b = nums[i], nums[j]
                remaining = [nums[k] for k in range(len(nums)) if k != i and k != j]

                # 定义运算
                # 注意：为了去重，我们在生成字符串时做特殊处理
                ops = []

                # 加法 (满足交换律)
                ops.append((a + b, "+", True))
                # 减法 (不满足交换律)
                ops.append((a - b, "-", False))
                # 乘法 (满足交换律)
                ops.append((a * b, "*", True))
                # 除法 (不满足交换律)
                if abs(b) > 1e-5:
                    ops.append((a / b, "/", False))

                for res, op_sym, is_commutative in ops:
                    # --- 🔥 关键去重逻辑：规范化算式文本 ---
                    # 如果是加法或乘法，强制把小的数放在前面
                    # 例如：遇到 5 + 1，我们记录为 "1 + 5"
                    if is_commutative and a > b:
                        step_str = f"{self._fmt(b)} {op_sym} {self._fmt(a)} = {self._fmt(res)}"
                    else:
                        step_str = f"{self._fmt(a)} {op_sym} {self._fmt(b)} = {self._fmt(res)}"

                    new_nums = remaining + [res]
                    new_history = node.history + [step_str]
                    next_nodes.append(ToTNode(new_nums, new_history, parent=node))
        return next_nodes

    async def _ask_llm_for_score(self, nums: List[float]) -> float:
        """
        三数时采用评分机制，这里符合原论文的设置
        """
        numbers_str = ', '.join([self._fmt(n) for n in nums])
        messages = [
            {"role": "system", "content": "You are a Game of 24 expert."},
            {"role": "user", "content": (
                f"Analyze if {numbers_str} can make 24.\n"
                "Look for patterns like:\n"
                "- (A * B) + C = 24\n"
                "- 32 - 8 = 24 logic\n"
                "Reply strictly: 'Sure', 'Likely', 'Impossible'."
            )}
        ]
        answer = await self.think(messages, temperature=0.7) # 0.7保证评分多样性，完成错漏答案
        answer = answer.lower()
        if "sure" in answer:
            return 20.0
        elif "likely" in answer:
            return 1.0
        elif "impossible" in answer:
            return 0.001
        return 0.5

    async def _evaluate_node(self, node: ToTNode) -> float:
        nums = node.current_numbers
        # 终局
        if node.is_terminal:
            return 100.0 if abs(nums[0] - 24.0) < 1e-5 else 0.0

        # 剩2数 (机器接管)
        # 注意：这里我们不做 history 修改，只负责打分，具体的算式生成交给下一轮的 _get_next_steps
        # 这样才能保证最后一步也能被规范化处理
        if len(nums) == 2:
            a, b = nums[0], nums[1]
            possibles = {a + b, a - b, b - a, a * b}
            if abs(b) > 1e-5: possibles.add(a / b)
            if abs(a) > 1e-5: possibles.add(b / a)
            for val in possibles:
                if abs(val - 24.0) < 1e-5: return 100.0
            return 0.001

        # 3数及以上 (LLM)
        return await self._ask_llm_for_score(nums)

    async def solve_all(self, initial_numbers: List[float]) -> List[List[str]]:
        print(f"🤖 ToTAgent 正在寻找最佳解法: {initial_numbers}")

        root = ToTNode(initial_numbers, [])
        current_layer = [root]
        found_solutions = []

        for step in range(3):
            print(f"\n--- 步骤 {step + 1} ---")

            candidates = []
            for node in current_layer:
                candidates.extend(self._get_next_steps(node))

            print(f"生成了 {len(candidates)} 种候选数字组.")

            tasks = [self._evaluate_node(node) for node in candidates]
            scores = await asyncio.gather(*tasks)

            for node, score in zip(candidates, scores):
                node.value = score

            valid_candidates = [n for n in candidates if n.value > 0.0001]
            valid_candidates.sort(key=lambda x: x.value, reverse=True)

            current_layer = valid_candidates[:self.beam_width]
            print(f"已经收集 {len(current_layer)} 种途径")

        # --- 最终收集与去重 ---
        unique_solution_strs = set()

        for node in current_layer:
            if node.is_terminal and abs(node.current_numbers[0] - 24.0) < 1e-5:
                # 将 history 列表拼接成字符串，作为唯一指纹
                sol_str = " | ".join(node.history)

                # 这里的 sol_str 已经是规范化过的（因为 _get_next_steps 做了排序）
                # 所以 1+5 和 5+1 在这里是完全一样的字符串
                if sol_str not in unique_solution_strs:
                    unique_solution_strs.add(sol_str)
                    found_solutions.append(node.history)

        return found_solutions


if __name__ == "__main__":
    async def main():
        agent = ToTAgent(model="gpt-4o-mini", beam_width=100)

        # 测试用例
        test_case = [0,0,0,0]

        solutions = await agent.solve_all(test_case)

        if solutions:
            print(f"\n✨ 发现 {len(solutions)} 种不同的解法 {test_case}:")
            for i, sol in enumerate(solutions):
                print(f"解法 {i + 1}: " + "  ->  ".join(sol))
        else:
            print("\n❌ 该算式无解法。")


    asyncio.run(main())
