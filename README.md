# Policy-Aware LLM Agents for Regulated Customer Service Workflows

这是论文公开 artifact 的代码发布包，只保留可公开的主 benchmark 评测代码。

## 包含内容
- `scripts/policy_workflow_benchmark_public.py`：公开版主 benchmark 脚本
- `tests/test_policy_workflow_benchmark_public.py`：对应测试
- `requirements.txt`：最小 Python 依赖

## 不包含内容
- 模型服务代码
- 内部 workflow runtime 代码
- 生图脚本、PPT 生成脚本、图后处理脚本
- 内部证据文件名、内部证据原文、生产日志

## 说明
这个公开版脚本保留了论文主 benchmark 的生成与评测逻辑，但移除了内部证据引用字段，适合公开代码仓库与匿名 artifact 提交。

## 运行测试
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
