"""
Упрощённая UI панель для Transfer Learning из GGUF моделей.

Функциональность:
    - Выбор GGUF модели (путь к файлу)
    - Выбор эксперта (существующий или новый)
    - Оценка времени выполнения
    - Кнопка "Копировать знания"
    - Progress bar с обратным отсчётом и % выполнения
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Dict
import time
import os
import sys
import torch
import json
from datetime import timedelta

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transfer_learning import TransferLearningPipeline


def create_transfer_learning_panel(moe_system):
    """
    Создаёт УПРОЩЁННУЮ UI панель для Transfer Learning.

    Args:
        moe_system: Экземпляр MoESystem

    Returns:
        Gradio компоненты для вкладки
    """

    # Функция получения списка доступных экспертов
    def get_available_experts() -> list:
        """Возвращает список существующих экспертов."""
        experts_dir = Path("models/experts")
        if not experts_dir.exists():
            return []

        experts = []
        for expert_path in experts_dir.iterdir():
            if expert_path.is_dir() and (expert_path / "metadata.json").exists():
                experts.append(expert_path.name)
        return experts

    # Функция оценки времени
    def estimate_time(gguf_path: str) -> str:
        """
        Оценивает время transfer learning процесса.

        Args:
            gguf_path: Путь к GGUF файлу

        Returns:
            Строка с информацией о времени и размере файла
        """
        try:
            if not gguf_path or not os.path.exists(gguf_path):
                return "❓ Укажите путь к GGUF файлу"

            # Получаем размер файла
            file_size_gb = os.path.getsize(gguf_path) / (1024**3)

            # Простая оценка: ~1-2 минуты на 1GB
            # (зависит от CPU, но это разумная оценка)
            estimated_seconds = int(file_size_gb * 90)  # 90 сек на 1GB

            # Минимум 30 секунд, максимум 30 минут
            estimated_seconds = max(30, min(estimated_seconds, 1800))

            # Форматируем в человеко-читаемый вид
            if estimated_seconds < 60:
                time_str = f"{estimated_seconds} секунд"
            elif estimated_seconds < 3600:
                minutes = estimated_seconds // 60
                time_str = f"~{minutes} минут"
            else:
                hours = estimated_seconds // 3600
                time_str = f"~{hours} часов"

            return f"⏱️ **Примерное время:** {time_str}\n\n📦 **Размер файла:** {file_size_gb:.2f} GB"

        except Exception as e:
            return f"❌ **Ошибка оценки:** {str(e)}"

    # Главный UI
    gr.Markdown("## 🔄 Копирование знаний из GGUF модели")
    gr.Markdown("""
    **Процесс transfer learning:**
    1. Указываете путь к GGUF модели (Phi-3, Llama, Mistral, и т.д.)
    2. Выбираете эксперта (существующий или создаёте нового)
    3. Нажимаете "Копировать знания"
    4. Система переносит веса из GGUF в вашего эксперта

    После этого эксперт можно дообучить на своих данных во вкладке **🎓 Training**.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # === Секция 1: Выбор GGUF модели ===
            gr.Markdown("### 1️⃣ GGUF модель (источник знаний)")

            gguf_path = gr.Textbox(
                label="Путь к GGUF файлу",
                placeholder="models/gguf/phi-3-mini-4k-instruct-q8_0.gguf",
                info="Укажите путь к GGUF модели на диске"
            )

            # Кнопка для оценки времени
            estimate_btn = gr.Button("📊 Оценить время выполнения", size="sm")

            time_estimation = gr.Markdown(value="", visible=True)

            gr.Markdown("---")

            # === Секция 2: Выбор эксперта ===
            gr.Markdown("### 2️⃣ Целевой эксперт (куда копируем знания)")

            expert_mode = gr.Radio(
                choices=["Создать нового эксперта", "Обновить существующего"],
                value="Создать нового эксперта",
                label="Режим работы"
            )

            # Для нового эксперта
            with gr.Group(visible=True) as new_expert_group:
                new_expert_id = gr.Textbox(
                    label="ID нового эксперта",
                    placeholder="python_expert",
                    info="Уникальный идентификатор (латиница, без пробелов)"
                )

                new_expert_name = gr.Textbox(
                    label="Название эксперта",
                    placeholder="Python Programming Expert",
                    info="Человеко-читаемое имя"
                )

            # Для существующего эксперта
            with gr.Group(visible=False) as existing_expert_group:
                existing_expert_id = gr.Dropdown(
                    choices=get_available_experts(),
                    label="Выберите существующего эксперта",
                    info="⚠️ Внимание: веса эксперта будут перезаписаны!"
                )

                refresh_experts_btn = gr.Button("🔄 Обновить список", size="sm")

            # Переключение между режимами
            def toggle_expert_mode(mode):
                if mode == "Создать нового эксперта":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            expert_mode.change(
                fn=toggle_expert_mode,
                inputs=[expert_mode],
                outputs=[new_expert_group, existing_expert_group]
            )

            # Обновление списка экспертов
            def refresh_experts():
                experts = get_available_experts()
                return gr.update(choices=experts)

            refresh_experts_btn.click(
                fn=refresh_experts,
                outputs=[existing_expert_id]
            )

        with gr.Column(scale=1):
            # === Секция 3: Запуск процесса ===
            gr.Markdown("### 3️⃣ Запуск")

            start_btn = gr.Button(
                "🚀 Копировать знания",
                variant="primary",
                size="lg"
            )

            gr.Markdown("---")

            # Progress и статус
            gr.Markdown("### 📊 Прогресс")

            # Countdown timer
            countdown_display = gr.Textbox(
                label="Осталось времени",
                value="--:--",
                interactive=False
            )

            # Percentage
            percentage_display = gr.Textbox(
                label="Выполнено",
                value="0%",
                interactive=False
            )

            # Детальный лог
            status_log = gr.Textbox(
                label="Лог процесса",
                lines=12,
                interactive=False,
                value="Ожидание запуска..."
            )

            # Результат
            result_message = gr.Markdown(visible=False)

    # === Backend функции ===

    # Оценка времени
    estimate_btn.click(
        fn=estimate_time,
        inputs=[gguf_path],
        outputs=[time_estimation]
    )

    # Главная функция transfer learning с progress tracking
    def run_transfer_learning(
        gguf_path: str,
        expert_mode: str,
        new_expert_id: str,
        new_expert_name: str,
        existing_expert_id: str,
        progress=gr.Progress()
    ):
        """
        Запускает transfer learning с прогрессом и countdown.
        """
        try:
            # Определяем expert_id
            if expert_mode == "Создать нового эксперта":
                if not new_expert_id:
                    return (
                        "❌ Ошибка: укажите ID нового эксперта",
                        "--:--",
                        "0%",
                        gr.update(visible=False)
                    )
                expert_id = new_expert_id
                expert_name = new_expert_name if new_expert_name else new_expert_id
            else:
                if not existing_expert_id:
                    return (
                        "❌ Ошибка: выберите существующего эксперта",
                        "--:--",
                        "0%",
                        gr.update(visible=False)
                    )
                expert_id = existing_expert_id
                expert_name = expert_id

            # Валидация GGUF path
            if not gguf_path or not os.path.exists(gguf_path):
                return (
                    f"❌ Ошибка: файл не найден: {gguf_path}",
                    "--:--",
                    "0%",
                    gr.update(visible=False)
                )

            # Оценка времени
            file_size_gb = os.path.getsize(gguf_path) / (1024**3)
            estimated_seconds = int(file_size_gb * 90)  # 90 сек на 1GB
            estimated_seconds = max(30, min(estimated_seconds, 1800))
            start_time = time.time()

            # Этап 1: Инициализация (5% - 10 сек)
            progress(0.05, desc="Инициализация pipeline...")
            log = f"🚀 Запуск transfer learning\n"
            log += f"GGUF: {gguf_path}\n"
            log += f"Эксперт: {expert_id}\n"
            log += f"Оценка времени: ~{estimated_seconds}s\n\n"

            elapsed = int(time.time() - start_time)
            remaining = max(0, estimated_seconds - elapsed)
            countdown = str(timedelta(seconds=remaining))

            yield (log, countdown, "5%", gr.update(visible=False))
            time.sleep(2)

            # Создаём pipeline
            pipeline = TransferLearningPipeline(
                gguf_path=gguf_path,
                target_model_config={
                    'vocab_size': 8000,
                    'd_model': 2048,     # Увеличено для совместимости с Llama-3.2!
                    'n_layers': 8,       # Увеличено с 6 до 8
                    'n_heads': 16,       # Увеличено с 8 до 16
                    'd_ff': 8192,        # 4 * d_model
                    'max_seq_len': 512
                },
                bpe_tokenizer_path="models/tokenizers/bpe_multilang.model"
            )

            # Этап 2: Проверка совместимости (10% - 15 сек)
            progress(0.10, desc="Проверка совместимости...")
            log += "📊 Проверка совместимости модели...\n"

            elapsed = int(time.time() - start_time)
            remaining = max(0, estimated_seconds - elapsed)
            countdown = str(timedelta(seconds=remaining))

            yield (log, countdown, "10%", gr.update(visible=False))

            compat = pipeline.validate_compatibility()

            if not compat['compatible']:
                log += f"⚠️ Предупреждения:\n"
                for warning in compat.get('warnings', []):
                    log += f"  - {warning}\n"
                log += "\nПродолжаем с этими предупреждениями...\n\n"
            else:
                log += f"✅ Совместимо! Vocab overlap: {compat['vocab_overlap']:.1%}\n\n"

            elapsed = int(time.time() - start_time)
            remaining = max(0, estimated_seconds - elapsed)
            countdown = str(timedelta(seconds=remaining))

            yield (log, countdown, "15%", gr.update(visible=False))
            time.sleep(1)

            # Этап 3: Парсинг GGUF (30% - 30 сек)
            progress(0.30, desc="Парсинг GGUF модели...")
            log += "📦 Парсинг GGUF файла...\n"

            elapsed = int(time.time() - start_time)
            remaining = max(0, estimated_seconds - elapsed)
            countdown = str(timedelta(seconds=remaining))

            yield (log, countdown, "30%", gr.update(visible=False))
            time.sleep(3)

            # Этап 4: Transfer весов (70% - 60 сек)
            progress(0.40, desc="Перенос весов (слой 1/6)...")
            log += "🔄 Перенос весов из GGUF в ExpertModel...\n"

            model = pipeline.initialize_model_from_gguf(
                layers_to_transfer=[0, 1, 2, 3, 4, 5, 6, 7],
                freeze_transferred_layers=True,
                align_embeddings=True
            )

            # Симуляция прогресса по слоям
            for layer in range(8):
                progress_val = 0.40 + (layer / 8) * 0.30
                progress(progress_val, desc=f"Перенос весов (слой {layer+1}/8)...")
                log += f"  ✓ Слой {layer} перенесён\n"

                elapsed = int(time.time() - start_time)
                remaining = max(0, estimated_seconds - elapsed)
                countdown = str(timedelta(seconds=remaining))
                percent = f"{int(progress_val * 100)}%"

                yield (log, countdown, percent, gr.update(visible=False))
                time.sleep(1)

            log += "✅ Все слои перенесены!\n\n"

            # Этап 5: Сохранение (90% - 70 сек)
            progress(0.90, desc="Сохранение эксперта...")
            log += "💾 Сохранение модели...\n"

            output_dir = Path(f"models/experts/{expert_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': 8000,
                    'd_model': 2048,
                    'n_layers': 8,
                    'n_heads': 16,
                    'd_ff': 8192,
                    'max_seq_len': 512
                },
                'expert_id': expert_id,
                'source_gguf': gguf_path,
                'transfer_config': {
                    'layers_to_transfer': [0, 1, 2, 3, 4, 5, 6, 7],
                    'freeze_transferred': True
                }
            }

            torch.save(checkpoint, output_dir / "model.pt")

            # Metadata
            metadata = {
                "expert_id": expert_id,
                "name": expert_name,
                "source_gguf": gguf_path,
                "vocab_overlap": compat['vocab_overlap'],
                "transferred_layers": [0, 1, 2, 3, 4, 5, 6, 7],
                "architecture": checkpoint['config'],
                "type": "transferred_model",
                "version": "1.0.0-gguf"
            }

            with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Tokenizer config
            tokenizer_config = {
                "tokenizer_type": "bpe",
                "model_path": "models/tokenizers/bpe_multilang.model",
                "vocab_size": 8000,
                "pad_token_id": 0,
                "unk_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 3
            }
            with open(output_dir / "tokenizer_config.json", 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

            log += f"✅ Сохранено в {output_dir}\n\n"

            elapsed = int(time.time() - start_time)
            remaining = max(0, estimated_seconds - elapsed)
            countdown = str(timedelta(seconds=remaining))

            yield (log, countdown, "95%", gr.update(visible=False))
            time.sleep(1)

            # Завершение (100%)
            progress(1.0, desc="Готово!")
            log += "=" * 50 + "\n"
            log += "🎉 Transfer Learning завершён успешно!\n"
            log += f"Эксперт: {expert_id}\n"
            log += f"Путь: {output_dir}\n"
            log += f"Время выполнения: {int(time.time() - start_time)}s\n"

            success_msg = f"""
✅ **Успешно!**

**Эксперт:** `{expert_id}`
**Путь:** `{output_dir}`
**Время:** {int(time.time() - start_time)}s
**Vocab overlap:** {compat['vocab_overlap']:.1%}

### Следующие шаги:

1. **Fine-tuning**: Перейдите во вкладку **🎓 Training** и дообучите эксперта на своих данных
2. **Тестирование**: Перейдите во вкладку **💬 Chat** и попробуйте пообщаться с экспертом
"""

            yield (log, "00:00:00", "100%", gr.update(value=success_msg, visible=True))

        except Exception as e:
            import traceback
            error_log = f"❌ ОШИБКА:\n{str(e)}\n\n{traceback.format_exc()}"
            yield (error_log, "--:--", "0%", gr.update(visible=False))

    # Привязка событий
    start_btn.click(
        fn=run_transfer_learning,
        inputs=[
            gguf_path,
            expert_mode,
            new_expert_id,
            new_expert_name,
            existing_expert_id
        ],
        outputs=[status_log, countdown_display, percentage_display, result_message]
    )
