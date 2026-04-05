from core.agent import PaperReActAgent


def main():
    agent = PaperReActAgent()

    print("\n📚 论文智能助手 (ReAct Agent)")
    print("=" * 50)
    print("命令:")
    print("  upload <路径>  - 上传论文")
    print("  list          - 列出文件")
    print("  unload        - 卸载论文")
    print("  clear         - 清空所有")
    print("  history       - 查看对话历史")
    print("  quit          - 退出")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = input("Q: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            parts = user_input.split()
            cmd = parts[0].lower() if parts else ""

            if cmd == 'list':
                print(f"\n已上传文件: {agent.list_files()}\n")
            elif cmd == 'unload':
                result = agent.unload()
                print(f"\n{result}\n")
            elif cmd == 'clear':
                result = agent.clear_all()
                print(f"\n{result}\n")
            elif cmd == 'history':
                print("\n对话历史:")
                for msg in agent.conversation_history:
                    role = "用户" if msg["role"] == "user" else "助手"
                    print(f"  [{role}] {msg['content'][:50]}...")
                print()
            elif cmd == 'upload' and len(parts) > 1:
                file_path = user_input[len(cmd):].strip()
                file_path = file_path.strip('"\'')
                result = agent.upload(file_path)
                print(f"\n{result}\n")
            elif user_input:
                response = agent.chat(user_input)
                print(f"\n{response}\n")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n再见！")


if __name__ == "__main__":
    main()
