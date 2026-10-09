# 考核二：UART 串口通信

- 协议解析与回传源码：[`uart/uart_echo.py`](uart/uart_echo.py)
- 自动化协议测试：[`uart/test_protocol.py`](uart/test_protocol.py)
- 官方 Linux x86-64 虚拟串口测试器：[`virtual_serial_tester`](virtual_serial_tester)
- 官方测试与单测记录：[`uart/verification.log`](uart/verification.log)

协议实现只使用 Python 标准库，无需 pip 安装串口库。运行单测：

```bash
cd 02_UART/uart
python3 test_protocol.py -v
```

端到端回环需要一对虚拟串口；按 [`uart/README.md`](uart/README.md) 的步骤打开官方测试器，并在测试器等待连接时运行 `python3 uart_echo.py --count 4`。测试器是考核提供的独立工具，不是环境安装包。
