"""
Компонент Monitoring Panel для real-time мониторинга системы.

Функциональность:
    - Графики метрик (latency, throughput, memory)
    - Таблица активных экспертов
    - Статистика routing
    - System resources (CPU/RAM)
    - Auto-refresh каждые 2 секунды
"""

import gradio as gr
import pandas as pd
from typing import Dict, List, Any


def create_monitoring_panel(moe_system):
    """
    Создаёт панель мониторинга системы.

    Args:
        moe_system: Экземпляр MoESystem

    Returns:
        Gradio компоненты (размещённые внутри функции)
    """
    gr.Markdown("## Real-Time System Monitoring")
    gr.Markdown("Мониторинг производительности и использования ресурсов")

    # Системные метрики (числовые показатели)
    gr.Markdown("### 📈 Текущие показатели")

    with gr.Row():
        cpu_usage = gr.Number(
            label="CPU Usage (%)",
            interactive=False,
            value=0.0,
            precision=1
        )

        ram_usage = gr.Number(
            label="RAM Usage (MB)",
            interactive=False,
            value=0.0,
            precision=1
        )

        loaded_experts_count = gr.Number(
            label="Loaded Experts",
            interactive=False,
            value=0
        )

        avg_latency = gr.Number(
            label="Avg Latency (ms)",
            interactive=False,
            value=0.0,
            precision=2
        )

    # Графики метрик
    gr.Markdown("### 📊 Графики производительности")

    with gr.Row():
        # График latency
        latency_plot = gr.LinePlot(
            x="timestamp",
            y="latency_ms",
            title="Inference Latency (ms)",
            tooltip=["timestamp", "latency_ms"],
            x_title="Time",
            y_title="Latency (ms)"
        )

        # График throughput
        throughput_plot = gr.LinePlot(
            x="timestamp",
            y="throughput",
            title="Throughput (tokens/sec)",
            tooltip=["timestamp", "throughput"],
            x_title="Time",
            y_title="Tokens/sec"
        )

    with gr.Row():
        # График memory usage
        memory_plot = gr.BarPlot(
            x="module",
            y="memory_mb",
            title="Memory Usage by Module (MB)",
            tooltip=["module", "memory_mb"],
            x_title="Module",
            y_title="Memory (MB)"
        )

        # Статистика routing
        routing_stats_json = gr.JSON(
            label="Routing Statistics",
            value={}
        )

    # Таблица активных экспертов
    gr.Markdown("### 🤖 Active Experts")

    experts_table = gr.Dataframe(
        headers=["Expert ID", "Name", "Status", "Priority"],
        datatype=["str", "str", "str", "number"],
        interactive=False,
        wrap=True
    )

    # Data Flow таблица
    gr.Markdown("### 🔄 Data Flow Metrics")

    dataflow_table = gr.Dataframe(
        headers=["From → To", "Speed (KB/s)", "Activity"],
        datatype=["str", "number", "str"],
        interactive=False,
        wrap=True
    )

    # Кнопки управления
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh Now", variant="primary")
        auto_refresh_checkbox = gr.Checkbox(
            label="Auto-refresh (2s)",
            value=True,
            info="Автоматическое обновление каждые 2 секунды"
        )
        reset_metrics_btn = gr.Button("🗑️ Reset Metrics", variant="secondary")

    # === Helper Functions ===

    def update_all_metrics():
        """
        Обновляет все метрики и возвращает данные для UI.

        Returns:
            Tuple с данными для всех компонентов
        """
        try:
            # Получаем метрики от системы
            metrics = moe_system.get_system_metrics()

            # Текущие показатели
            cpu = metrics.get('cpu_usage_percent', 0.0)
            ram = metrics.get('memory_usage_mb', 0.0)
            loaded_count = metrics.get('loaded_experts_count', 0)
            avg_lat = metrics.get('latency_inference_avg_ms', 0.0)

            # История для графиков
            latency_history = moe_system.metrics_collector.get_history_for_plot(
                'latency_inference',
                max_points=50
            )

            throughput_history = moe_system.metrics_collector.get_history_for_plot(
                'throughput_tokens',
                max_points=50
            )

            # Форматируем для LinePlot
            latency_df = pd.DataFrame([
                {'timestamp': h['timestamp'], 'latency_ms': h['value']}
                for h in latency_history
            ]) if latency_history else pd.DataFrame(columns=['timestamp', 'latency_ms'])

            throughput_df = pd.DataFrame([
                {'timestamp': h['timestamp'], 'throughput': h['value']}
                for h in throughput_history
            ]) if throughput_history else pd.DataFrame(columns=['timestamp', 'throughput'])

            # Memory usage по модулям
            memory_data = []
            for module_id, stats in moe_system.module_stats.items():
                memory_data.append({
                    'module': module_id,
                    'memory_mb': stats.get('memory_mb', 0.0)
                })

            memory_df = pd.DataFrame(memory_data) if memory_data else pd.DataFrame(columns=['module', 'memory_mb'])

            # Routing statistics
            routing_stats = moe_system.metrics_collector.get_routing_statistics()

            # Таблица экспертов
            experts = moe_system.list_available_experts()
            experts_data = [
                [
                    e['id'],
                    e['name'],
                    e['status'],
                    e['priority']
                ]
                for e in experts
            ]

            # Data flow таблица
            data_flow = moe_system.get_data_flow_stats()
            dataflow_data = []

            for (from_mod, to_mod), speed in data_flow.items():
                speed_kb = speed / 1024  # bytes/sec → KB/sec

                # Определяем activity level
                if speed_kb > 100:
                    activity = "🟢 High"
                elif speed_kb > 10:
                    activity = "🟡 Medium"
                else:
                    activity = "🔵 Low"

                dataflow_data.append([
                    f"{from_mod} → {to_mod}",
                    round(speed_kb, 2),
                    activity
                ])

            return (
                cpu,
                ram,
                loaded_count,
                avg_lat,
                latency_df,
                throughput_df,
                memory_df,
                routing_stats,
                experts_data,
                dataflow_data
            )

        except Exception as e:
            print(f"Error updating metrics: {e}")

            # Возвращаем пустые данные
            return (
                0.0,
                0.0,
                0,
                0.0,
                pd.DataFrame(columns=['timestamp', 'latency_ms']),
                pd.DataFrame(columns=['timestamp', 'throughput']),
                pd.DataFrame(columns=['module', 'memory_mb']),
                {},
                [],
                []
            )

    def reset_metrics():
        """Сбрасывает все метрики."""
        moe_system.metrics_collector.reset()

        return update_all_metrics()

    # === Event Handlers ===

    # Ручное обновление
    refresh_btn.click(
        fn=update_all_metrics,
        outputs=[
            cpu_usage,
            ram_usage,
            loaded_experts_count,
            avg_latency,
            latency_plot,
            throughput_plot,
            memory_plot,
            routing_stats_json,
            experts_table,
            dataflow_table
        ]
    )

    # Сброс метрик
    reset_metrics_btn.click(
        fn=reset_metrics,
        outputs=[
            cpu_usage,
            ram_usage,
            loaded_experts_count,
            avg_latency,
            latency_plot,
            throughput_plot,
            memory_plot,
            routing_stats_json,
            experts_table,
            dataflow_table
        ]
    )

    # Auto-refresh
    timer = gr.Timer(2.0)

    def maybe_update(auto_enabled: bool):
        """Обновляет метрики только если auto-refresh включён."""
        if auto_enabled:
            return update_all_metrics()
        else:
            # Возвращаем gr.update() для сохранения текущих значений
            return tuple([gr.update()] * 10)

    timer.tick(
        fn=maybe_update,
        inputs=[auto_refresh_checkbox],
        outputs=[
            cpu_usage,
            ram_usage,
            loaded_experts_count,
            avg_latency,
            latency_plot,
            throughput_plot,
            memory_plot,
            routing_stats_json,
            experts_table,
            dataflow_table
        ]
    )

    # Инициализация при загрузке
    app_load_event = gr.on(
        triggers=[],  # Пустой список - загрузится при старте
        fn=update_all_metrics,
        outputs=[
            cpu_usage,
            ram_usage,
            loaded_experts_count,
            avg_latency,
            latency_plot,
            throughput_plot,
            memory_plot,
            routing_stats_json,
            experts_table,
            dataflow_table
        ]
    )
