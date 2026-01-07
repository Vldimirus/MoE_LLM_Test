#!/usr/bin/env python3
"""
Скрипт запуска Gradio Web UI Dashboard для MoE System.

Usage:
    python scripts/run_ui.py [--port 7860] [--share] [--config configs/ui_config.yaml]

Examples:
    # Запуск на порту 7860 (по умолчанию)
    python scripts/run_ui.py

    # Запуск с публичной ссылкой (Gradio share)
    python scripts/run_ui.py --share

    # Запуск на custom порту
    python scripts/run_ui.py --port 8080

    # Указание custom конфигурации
    python scripts/run_ui.py --config my_config.yaml
"""

import sys
import argparse
from pathlib import Path

# Добавляем src/python в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

from ui.app import create_gradio_app


def main():
    """Запускает Gradio приложение."""
    parser = argparse.ArgumentParser(
        description="Запуск MoE System Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ui.py
  python scripts/run_ui.py --port 8080
  python scripts/run_ui.py --share
  python scripts/run_ui.py --config my_config.yaml
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ui_config.yaml",
        help="Путь к файлу конфигурации (default: configs/ui_config.yaml)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Порт для UI (default: 7860)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Создать публичную ссылку через Gradio share"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки (детальные логи)"
    )

    args = parser.parse_args()

    # Печатаем информацию о запуске
    print("=" * 70)
    print("🤖 Domain-Specific MoE System Dashboard")
    print("=" * 70)
    print(f"📁 Config file: {args.config}")
    print(f"🌐 Server port: {args.port}")
    print(f"🔗 Public share: {'Yes' if args.share else 'No'}")
    print(f"🐛 Debug mode:   {'Yes' if args.debug else 'No'}")
    print("=" * 70)
    print()

    # Создаём и запускаем приложение
    try:
        print("🚀 Starting Gradio application...")
        print()

        app = create_gradio_app(config_path=args.config)

        print("✅ Application created successfully")
        print()

        # Информация для пользователя
        print("=" * 70)
        print(f"📊 Dashboard доступен по адресу: http://localhost:{args.port}")

        if args.share:
            print("🔗 Публичная ссылка будет сгенерирована Gradio...")

        print()
        print("💡 Tabs:")
        print("   💬 Chat          - Диалог с моделью")
        print("   📊 Monitoring    - Real-time метрики")
        print("   🏗️  Architecture - Визуализация системы")
        print("   🧪 Tests         - Запуск pytest")
        print("   🎓 Training      - Обучение моделей")
        print()
        print("⌨️  Для остановки: Ctrl+C")
        print("=" * 70)
        print()

        # Запуск
        app.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=not args.debug,
            favicon_path=None
        )

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⏹️  Dashboard остановлен пользователем")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Ошибка при запуске Dashboard: {str(e)}")
        print("=" * 70)

        if args.debug:
            import traceback
            print()
            print("Stack trace:")
            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
