"""
Главный Gradio Web UI Dashboard для Domain-Specific MoE System.

Предоставляет интерфейс с 5 основными табами:
    1. Chat - диалог с моделью через Router
    2. Monitoring - real-time системные метрики
    3. Architecture - динамическая визуализация архитектуры
    4. Tests - запуск pytest и просмотр результатов
    5. Training - обучение моделей с графиками прогресса
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr
from ui.backend.moe_system import MoESystem
from ui.components.chat_interface import create_chat_interface
from ui.components.monitoring_panel import create_monitoring_panel
from ui.components.architecture_viz import create_architecture_viz
from ui.components.test_panel import create_test_panel
from ui.components.training_panel import create_training_panel


def create_gradio_app(config_path: str = "configs/ui_config.yaml"):
    """
    Создаёт главный Gradio интерфейс с 5 табами.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Gradio Blocks приложение
    """
    # Инициализация MoE системы
    moe_system = MoESystem(config_path=config_path)

    # Создаём Gradio интерфейс
    with gr.Blocks(
        title="Domain-Specific MoE System Dashboard"
    ) as app:
        # Заголовок
        gr.Markdown(
            """
            # 🤖 Domain-Specific MoE System Dashboard

            Центр управления для тестирования, мониторинга и обучения Mixture of Experts системы
            """
        )

        # Информационная панель (краткая статистика)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Статус системы")
                system_status = gr.Textbox(
                    value="🟢 Система готова",
                    label="",
                    interactive=False,
                    show_label=False
                )

            with gr.Column(scale=1):
                gr.Markdown("### Загружено экспертов")
                loaded_experts = gr.Number(
                    value=0,
                    label="",
                    interactive=False,
                    show_label=False
                )

            with gr.Column(scale=1):
                gr.Markdown("### Использование памяти")
                memory_usage = gr.Textbox(
                    value="0 MB",
                    label="",
                    interactive=False,
                    show_label=False
                )

        gr.Markdown("---")

        # Основные табы
        with gr.Tabs() as tabs:
            # TAB 1: Chat Interface
            with gr.Tab("💬 Chat", id="chat"):
                create_chat_interface(moe_system)

            # TAB 2: System Monitoring
            with gr.Tab("📊 Monitoring", id="monitoring"):
                create_monitoring_panel(moe_system)

            # TAB 3: Architecture Visualization
            with gr.Tab("🏗️ Architecture", id="architecture"):
                create_architecture_viz(moe_system)

            # TAB 4: Test Runner
            with gr.Tab("🧪 Tests", id="tests"):
                create_test_panel(moe_system)

            # TAB 5: Training Dashboard
            with gr.Tab("🎓 Training", id="training"):
                create_training_panel(moe_system)

        # Футер
        gr.Markdown(
            """
            ---

            **Domain-Specific MoE System** | Версия 0.4.0 |
            [Документация](docs/) | [GitHub](https://github.com/)
            """
        )

        # Периодическое обновление статуса системы
        def update_system_status():
            """Обновляет краткую статистику в header."""
            metrics = moe_system.get_system_metrics()

            loaded_count = metrics.get('loaded_experts_count', 0)
            memory_mb = metrics.get('memory_usage_mb', 0.0)

            return (
                "🟢 Система активна",
                loaded_count,
                f"{memory_mb:.1f} MB"
            )

        # Auto-refresh статуса каждые 5 секунд
        status_timer = gr.Timer(5.0)
        status_timer.tick(
            fn=update_system_status,
            outputs=[system_status, loaded_experts, memory_usage]
        )

    return app


def main():
    """Запускает Gradio приложение."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Запуск MoE System Dashboard"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ui_config.yaml",
        help="Путь к файлу конфигурации"
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
        help="Создать публичную ссылку (Gradio share)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Domain-Specific MoE System Dashboard")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("=" * 70)

    # Создаём и запускаем приложение
    app = create_gradio_app(config_path=args.config)

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
