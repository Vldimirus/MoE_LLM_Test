"""
Компонент Chat Interface для диалога с MoE системой.

Функциональность:
    - Chatbot с историей диалога
    - Выбор режима: Auto Router или Manual expert selection
    - Боковая панель с метриками последнего ответа
    - Очистка истории
"""

import gradio as gr
from typing import List, Tuple, Any


def create_chat_interface(moe_system):
    """
    Создаёт интерфейс для чата с моделью.

    Args:
        moe_system: Экземпляр MoESystem

    Returns:
        Gradio компоненты (размещённые внутри функции)
    """
    gr.Markdown("## Диалог с MoE системой")
    gr.Markdown(
        "Общайтесь с AI через автоматический Router или выбирайте эксперта вручную"
    )

    with gr.Row():
        # Основная колонка с чатом
        with gr.Column(scale=3):
            # Chatbot
            chatbot = gr.Chatbot(
                label="История диалога",
                height=500,
                avatar_images=(
                    None,  # User avatar
                    None   # Bot avatar
                )
            )

            # Ввод сообщения
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Введите ваш вопрос или команду...",
                    scale=4,
                    lines=2,
                    max_lines=5,
                    show_label=False
                )
                send_btn = gr.Button(
                    "▶️ Отправить",
                    scale=1,
                    variant="primary",
                    size="lg"
                )

            # Настройки режима
            with gr.Row():
                with gr.Column(scale=1):
                    expert_mode = gr.Radio(
                        choices=["Auto (Router)", "Manual"],
                        value="Auto (Router)",
                        label="Режим выбора эксперта",
                        info="Auto использует Router для автоматического выбора"
                    )

                with gr.Column(scale=1):
                    # Получаем список экспертов
                    experts = moe_system.list_available_experts()
                    expert_choices = [e['id'] for e in experts]

                    expert_dropdown = gr.Dropdown(
                        choices=expert_choices,
                        label="Выбор эксперта (Manual режим)",
                        value=expert_choices[0] if expert_choices else None,
                        visible=False,
                        interactive=True
                    )

            # Дополнительные параметры генерации
            with gr.Accordion("⚙️ Параметры генерации", open=False):
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Tokens",
                        info="Максимальное количество токенов для генерации"
                    )

                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Креативность ответа (0=детерминированный, 2=очень креативный)"
                    )

            # Кнопка очистки
            clear_btn = gr.Button("🗑️ Очистить историю", variant="secondary")

        # Боковая панель с метриками
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Метрики последнего ответа")

            expert_used = gr.Textbox(
                label="Использованный эксперт",
                interactive=False,
                value="-"
            )

            routing_confidence = gr.Number(
                label="Confidence (%)",
                interactive=False,
                value=0.0,
                precision=1
            )

            latency = gr.Number(
                label="Latency (ms)",
                interactive=False,
                value=0.0,
                precision=2
            )

            tokens_generated = gr.Number(
                label="Токенов сгенерировано",
                interactive=False,
                value=0
            )

            gr.Markdown("### 💾 Статистика памяти")

            memory_stats = gr.JSON(
                label="Memory Levels",
                value={}
            )

            gr.Markdown("### ℹ️ Подсказки")

            gr.Markdown(
                """
                **Примеры запросов:**

                - "Напиши функцию на Python для сортировки списка"
                - "Реши уравнение: x² - 5x + 6 = 0"
                - "Объясни что такое машинное обучение"
                """
            )

    # === Event Handlers ===

    def respond(
        message: str,
        history: List[Tuple[str, str]],
        mode: str,
        expert_id: str,
        max_tok: int,
        temp: float
    ):
        """
        Обрабатывает сообщение пользователя и возвращает ответ.

        Args:
            message: Сообщение пользователя
            history: История диалога
            mode: Режим ('Auto (Router)' или 'Manual')
            expert_id: ID эксперта (для Manual режима)
            max_tok: Максимум токенов
            temp: Temperature

        Returns:
            Обновлённые значения для UI компонентов
        """
        if not message.strip():
            return (
                history,
                "",  # Очищаем input
                expert_used.value,
                routing_confidence.value,
                latency.value,
                tokens_generated.value,
                memory_stats.value
            )

        # Определяем expert_id для запроса
        use_expert_id = None if mode == "Auto (Router)" else expert_id

        # Вызываем MoE систему
        try:
            response = moe_system.chat(
                user_message=message,
                expert_id=use_expert_id,
                max_tokens=max_tok,
                temperature=temp
            )

            # Добавляем в историю
            history.append((message, response['response']))

            return (
                history,
                "",  # Очищаем input
                response['expert_used'],
                response['routing_confidence'] * 100,  # В проценты
                response['latency_ms'],
                response['tokens_generated'],
                response['memory_stats']
            )

        except Exception as e:
            error_msg = f"Ошибка: {str(e)}"
            history.append((message, error_msg))

            return (
                history,
                "",
                "error",
                0.0,
                0.0,
                0,
                {}
            )

    def clear_history():
        """Очищает историю диалога."""
        return (
            [],  # Пустая история
            "-",
            0.0,
            0.0,
            0,
            {}
        )

    def toggle_expert_dropdown(mode: str):
        """Показывает/скрывает dropdown экспертов в зависимости от режима."""
        return gr.update(visible=(mode == "Manual"))

    # Подключаем обработчики

    # Отправка сообщения (по кнопке)
    send_btn.click(
        fn=respond,
        inputs=[
            msg_input,
            chatbot,
            expert_mode,
            expert_dropdown,
            max_tokens,
            temperature
        ],
        outputs=[
            chatbot,
            msg_input,
            expert_used,
            routing_confidence,
            latency,
            tokens_generated,
            memory_stats
        ]
    )

    # Отправка сообщения (по Enter)
    msg_input.submit(
        fn=respond,
        inputs=[
            msg_input,
            chatbot,
            expert_mode,
            expert_dropdown,
            max_tokens,
            temperature
        ],
        outputs=[
            chatbot,
            msg_input,
            expert_used,
            routing_confidence,
            latency,
            tokens_generated,
            memory_stats
        ]
    )

    # Очистка истории
    clear_btn.click(
        fn=clear_history,
        outputs=[
            chatbot,
            expert_used,
            routing_confidence,
            latency,
            tokens_generated,
            memory_stats
        ]
    )

    # Переключение видимости dropdown
    expert_mode.change(
        fn=toggle_expert_dropdown,
        inputs=[expert_mode],
        outputs=[expert_dropdown]
    )
