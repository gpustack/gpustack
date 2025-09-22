# 开发指南

## 先决条件

安装 Python（版本 3.10 至 3.12）。

## 设置环境

```bash
make install
```

## 运行

```bash
poetry run gpustack
```

## 构建

```bash
make build
```

并在 `dist` 中检查构建产物。

## 测试

```bash
make test
```

## 更新依赖

```bash
poetry add <something>
```

或

```bash
poetry add --group dev <something>
```

用于开发/测试依赖。