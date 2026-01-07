"""
Компонент Test Panel для запуска pytest и просмотра результатов.

Функциональность:
    - Выбор тестов для запуска
    - Запуск pytest с coverage
    - Real-time прогресс
    - Результаты: passed/failed/skipped
    - Coverage отчёт по модулям
    - Детали провалившихся тестов
"""

import gradio as gr
import pandas as pd
from ui.backend.test_runner import TestRunner


def create_test_panel(moe_system):
    """
    Создаёт панель для запуска тестов.

    Args:
        moe_system: Экземпляр MoESystem

    Returns:
        Gradio компоненты (размещённые внутри функции)
    """
    # Инициализация TestRunner
    test_runner = TestRunner(tests_dir="tests/")

    gr.Markdown("## Test Runner")
    gr.Markdown("Запуск pytest с автоматическим coverage анализом")

    # Настройки запуска
    gr.Markdown("### ⚙️ Настройки")

    with gr.Row():
        # Выбор тестов
        test_list = test_runner.get_test_list()
        test_choices = [t['name'] for t in test_list]
        test_paths = {t['name']: t['path'] for t in test_list}

        test_selection = gr.Dropdown(
            choices=test_choices,
            value=test_choices[0] if test_choices else None,
            label="Выбор тестов",
            info="Выберите набор тестов для запуска",
            scale=2
        )

        with_coverage = gr.Checkbox(
            label="Включить Coverage анализ",
            value=True,
            info="Измерение покрытия кода тестами",
            scale=1
        )

        verbose = gr.Checkbox(
            label="Verbose output",
            value=True,
            info="Детальный вывод pytest",
            scale=1
        )

    # Кнопка запуска
    run_btn = gr.Button("▶️ Run Tests", variant="primary", size="lg")

    # Прогресс
    gr.Markdown("### 📊 Прогресс выполнения")

    progress_text = gr.Textbox(
        label="Статус",
        value="Готов к запуску тестов",
        interactive=False
    )

    # Результаты
    gr.Markdown("### 📈 Результаты")

    with gr.Row():
        total_tests = gr.Number(
            label="Всего тестов",
            interactive=False,
            value=0
        )

        passed = gr.Number(
            label="✅ Passed",
            interactive=False,
            value=0
        )

        failed = gr.Number(
            label="❌ Failed",
            interactive=False,
            value=0
        )

        skipped = gr.Number(
            label="⏭️ Skipped",
            interactive=False,
            value=0
        )

        duration = gr.Number(
            label="⏱️ Duration (sec)",
            interactive=False,
            value=0.0,
            precision=2
        )

    # Статистика успеха
    with gr.Row():
        success_rate = gr.Number(
            label="Success Rate (%)",
            interactive=False,
            value=0.0,
            precision=1
        )

        coverage_pct = gr.Number(
            label="Coverage (%)",
            interactive=False,
            value=0.0,
            precision=1
        )

    # Coverage report по модулям
    gr.Markdown("### 📊 Coverage Report by Module")

    coverage_plot = gr.BarPlot(
        x="module",
        y="coverage_pct",
        title="Test Coverage by Module (%)",
        x_title="Module",
        y_title="Coverage (%)",
        tooltip=["module", "coverage_pct"],
        color="coverage_pct"
    )

    # Детали провалившихся тестов
    gr.Markdown("### ❌ Failed Tests Details")

    failed_tests_table = gr.Dataframe(
        headers=["Test Name", "Location", "Error"],
        datatype=["str", "str", "str"],
        interactive=False,
        wrap=True,
        column_widths=["30%", "30%", "40%"]
    )

    # === Helper Functions ===

    def run_tests_handler(
        test_name: str,
        with_cov: bool,
        verbose_flag: bool,
        progress=gr.Progress()
    ):
        """
        Запускает тесты и возвращает результаты.

        Args:
            test_name: Название выбранного набора тестов
            with_cov: Включить coverage
            verbose_flag: Verbose output
            progress: Gradio Progress tracker

        Returns:
            Tuple с результатами для UI компонентов
        """
        try:
            # Получаем путь к тестам
            test_pattern = test_paths.get(test_name, "test_*.py")

            # Обновляем статус
            progress(0, desc="Запуск тестов...")

            # Callback для прогресса (упрощённый, так как pytest не даёт real-time)
            def progress_callback(current, total):
                progress((current, total), desc=f"Выполнение тестов... {current}/{total}")

            # Запускаем тесты
            results = test_runner.run_tests(
                test_pattern=test_pattern,
                with_coverage=with_cov,
                callback=progress_callback,
                verbose=verbose_flag
            )

            # Извлекаем результаты
            total = results.get('total_tests', 0)
            pass_count = results.get('passed', 0)
            fail_count = results.get('failed', 0)
            skip_count = results.get('skipped', 0)
            dur = results.get('duration_sec', 0.0)
            cov = results.get('coverage_pct', 0.0)

            # Вычисляем success rate
            success = (pass_count / total * 100) if total > 0 else 0.0

            # Статус
            if fail_count == 0:
                status = f"✅ Все тесты прошли успешно! ({total} тестов за {dur:.2f}с)"
            else:
                status = f"❌ {fail_count} тест(ов) провалилось из {total}"

            # Coverage report
            coverage_data = []
            if with_cov:
                coverage_report = test_runner.get_coverage_report()

                for module, pct in coverage_report.items():
                    # Берём только src/python файлы
                    if 'src/python' in module:
                        # Упрощаем имя модуля
                        short_name = module.replace('src/python/', '')

                        coverage_data.append({
                            'module': short_name,
                            'coverage_pct': pct
                        })

            coverage_df = pd.DataFrame(coverage_data) if coverage_data else pd.DataFrame(columns=['module', 'coverage_pct'])

            # Failed tests детали
            failed_tests = results.get('failed_tests', [])
            failed_data = [
                [test['name'], test['location'], test['error'][:200]]
                for test in failed_tests
            ]

            return (
                status,
                total,
                pass_count,
                fail_count,
                skip_count,
                dur,
                success,
                cov,
                coverage_df,
                failed_data
            )

        except Exception as e:
            error_msg = f"Ошибка при запуске тестов: {str(e)}"

            return (
                error_msg,
                0,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                pd.DataFrame(columns=['module', 'coverage_pct']),
                []
            )

    # === Event Handlers ===

    # Запуск тестов
    run_btn.click(
        fn=run_tests_handler,
        inputs=[test_selection, with_coverage, verbose],
        outputs=[
            progress_text,
            total_tests,
            passed,
            failed,
            skipped,
            duration,
            success_rate,
            coverage_pct,
            coverage_plot,
            failed_tests_table
        ]
    )

    # Информация о тестах
    gr.Markdown(
        """
        ### ℹ️ Информация

        **Доступные тесты:**
        - `Все тесты` - запускает весь test suite (211 тестов)
        - `Transformer tests` - тесты для MultiHeadAttention, FeedForward, TransformerBlock
        - `Expert tests` - тесты для ExpertModel и генерации
        - `Router tests` - тесты для SimpleRouter
        - `Memory tests` - тесты для ThreeLevelMemory
        - `Dataset tests` - тесты для Tokenizer и Dataset
        - `Trainer tests` - тесты для training pipeline

        **Coverage:**
        - Показывает процент кода, покрытого тестами
        - Целевой показатель: >85% для критических модулей
        """
    )
