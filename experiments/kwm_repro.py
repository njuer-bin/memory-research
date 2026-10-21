import time, sys
sys.path.insert(0, ".")
import kwm_common as C

sys_p = ("你是数据分析助手。你只能用工具完成任务，禁止直接编造数字。\n"
         "可用工具：run_python(代码)。\n"
         "当你要调用工具时，严格只输出以下格式（不要任何额外文字）：\n"
         "ACTION: run_python\n"
         "<python 代码>\n"
         "当你完成计算并得到真实结果后，输出：\n"
         "FINAL:\n"
         "<中文结论，含真实数值>\n")

user = ("请调用 run_python 读取工作目录下的 train_v1.csv（列 group,z1,z2,y），"
        "删除 y 为空的行后用 pandas 计算 A 组与 B 组 y 的均值差 Δ=A.mean()-B.mean()，"
        "并 print('DELTA', delta, 'N', n)。只输出 ACTION 块。")

msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
t0 = time.time()
r = C.chat(msgs, timeout=90)
print("chat=", round(time.time()-t0,2), "s")
print("CONTENT:\n", r.get("message",{}).get("content"))
