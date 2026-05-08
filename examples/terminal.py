from tools.terminal_tool import TerminalTool

if __name__ == "__main__":
    terminal_tool = TerminalTool(workspace="./")

    print(terminal_tool.run({"command": "ls -l"}))
