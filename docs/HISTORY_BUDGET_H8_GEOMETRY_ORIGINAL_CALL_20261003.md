# H8几何原调用零步取证

2026-10-03。H2几何原队列完整完成且退出后，空闲GPU1按原卡锁和连续60秒资源门禁执行了已准备的独立取证工具。固定诊断提交为`12a920cdd1aab23d9d99eee48bac03c31cf0efce`，科学提交仍`0f33492d1b639da716897d8faa4a1df293354c49`，两者文件SHA分别绑定；原失败claim和失败状态保持不变。

四个原anchor（0/1024/2048/4096）的原`gpu_contract`全部完成。两模型每anchor各一次原native推理之后、原13张量与精确体素断言之前，仅插入CPU持久化回调；移除此唯一回调会精确恢复原AST。原语句、调用顺序、容差与失败传播保持，没有额外模型/get_occ/deterministic调用，没有优化器或优化步骤，也没有执行新的正式准入、smoke或训练。

## 冻结证据独立核验

约377.70MB的四份原Torch张量文件、四份native体素文件、原报告、启动收据及日志已下载；压缩归档11个原文件逐字节大小/SHA通过。该归档只含本次取证数据，不能补回原失败时第三个anchor未保存的张量。

本机使用受限pickle白名单和Numpy独立读回Torch zip的CPU存储、offset、shape、stride及dtype。生产端字节序证据为little，旧序列化文件未内嵌byteorder字段，明确采用该生产端证据，不猜测存储格式。四anchor各13对张量schema相同、数值有限且逐值精确；四时域native坐标与类别同样逐值精确。没有为核验额外执行模型或GPU后处理。

| anchor | 原张量对 | raw有限且精确 | 四时域坐标/类别精确 |
|---:|---:|---|---|
| 0 | 13 | 是 | 是 |
| 1024 | 13 | 是 | 是 |
| 2048 | 13 | 是 | 是 |
| 4096 | 13 | 是 | 是 |

公开摘要保留全部张量shape/dtype及原SHA；四份native体素为原文件直接复制。约376.58MB的raw张量、含账户/进程信息的原收据和日志仅私有归档。公开材料剔除账户、PID、命令与进程路径。

- [独立读回摘要与SHA](evidence/history_budget_h8_geometry_original_call_20261003/verification.json)
- [四份原native体素](evidence/history_budget_h8_geometry_original_call_20261003/)
- [固定原调用取证工具](../tools/diagnose_budget_original_call_parity.py)

## 结论与成本边界

本次未复现原H8几何第三个anchor当前体素坐标差异，根因仍未知。原失败现场缺少该anchor冻结张量，无法用本次不同时间的精确结果替代当时证据。CPU存盘还改变anchor之间的时间，不能据通过认定`deterministic=False`为根因、正式准入通过或自动恢复失败训练；原失败记录和原准入标准均保持。一次独立完整重新准入是否执行，等待用户对原“明确实现错误证据才修复”限制的决定。

本次零优化取证生命周期的保守下/上界为116.886698–133.698478秒：用Linux整数秒btime和starttime/HZ估计创建时间并保留1秒分辨率；下界截止进程内finished标记，上界截止首次确认/proc消失，退出观测区间为15.811780秒。该进程不是本地可waitpid子进程，退出码保持null，不伪造0。范围包含60.247450秒资源门禁、初始化、推理和CPU存盘及内部等待；各anchor存盘elapsed已嵌套其中，不再相加。这项诊断成本与正式训练及部署benchmark分列，不当延迟、显存或能耗收益。

H2几何的完整结果另见[阶段报告](HISTORY_BUDGET_H2_GEOMETRY_COMPLETE_20261003.md)。四臂bootstrap、非劣/交互及成本benchmark仍等待完整主线结果；没有提前四组结论。
